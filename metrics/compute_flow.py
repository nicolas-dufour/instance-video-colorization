#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np
from collections import namedtuple

### torch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2

### custom lib
from models.raft.raft import RAFT
import utils
def warp(img, flow):
    
    return cv2.remap(img, flow, None, cv2.INTER_LINEAR)

class Args:
    pass
def compute_warp_error(video_path):
    args = argparse.Namespace(small=False, mixed_precision=False)
    ### initialize FlowNet
    model = torch.nn.DataParallel(RAFT(args))

    ### load pre-trained FlowNet
    model_filename = os.path.join("pretrained_models", "RAFT_checkpoint.pth")
    print("===> Load %s" %model_filename)
    model.load_state_dict(torch.load(model_filename))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

  
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    frame_list = [image]
    success = True
    while success:
        success,image = vidcap.read()
        frame_list.append(image)

    warping_error = 0
    for t in range(0,len(frame_list)-2):
       ### load input images 
        img1 = frame_list[t]
        img2 = frame_list[t+1]

        if(img2 is None):
            print(t)
        
        ### resize image
        size_multiplier = 64
        H_orig = img1.shape[0]
        W_orig = img1.shape[1]

        H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
        W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)
        
        img1_r = cv2.resize(img1, (W_sc, H_sc))
        img2_r = cv2.resize(img2, (W_sc, H_sc))
    
        with torch.no_grad():

            ### convert to tensor
            img1_t = utils.img2tensor(img1_r).to(device)
            img2_t = utils.img2tensor(img2_r).to(device)
    
            ### compute fw flow
            fw_flow = model(img1_t, img2_t, test_mode=True)[1]
            fw_flow = utils.tensor2img(fw_flow)
        
            ### compute bw flow
            bw_flow = model(img1_t, img2_t, test_mode=True)[1]
            bw_flow = utils.tensor2img(bw_flow)


        ### resize flow
        fw_flow = utils.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig) 
        bw_flow = utils.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig) 
        ### compute occlusion
        fw_occ = utils.detect_occlusion(bw_flow, fw_flow)
        noc_mask = 1 - fw_occ
        warp_img1 = warp(img1,fw_flow)

        diff = noc_mask[:,:,np.newaxis]*np.abs(img2-warp_img1)

        N = noc_mask.sum()
        if N==0:
            N=img2.shape[0]*img2.shape[1]
        warping_error += diff.sum()/N
    warping_error = warping_error/(len(frame_list)-1)
    return warping_error
        
        