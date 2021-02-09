#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2

### custom lib
import networks
from networks.FlowNet2 import FlowNet2
from networks.resample2d_package.modules.resample2d import Resample2d
import utils



def compute_warp_error(video_path)

    flow_options = {}
    flow_options.rgb_max = 1.0
    flow_options.fp16 = False

    print(opts)

      
    ### initialize FlowNet
    print('===> Initializing model from %s...' %opts.model)
    model = FlowNet2(flow_options)

    ### load pre-trained FlowNet
    model_filename = os.path.join("pretrained_models", "FlowNet2_checkpoint.pth.tar")
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    flow_warping = Resample2d().to(device)
  
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    frame_list = [image]
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        frame_list.append(image)
        count += 1

    warping_error = 0
    for t in range(1,len(frame_list)):
        
       ### load input images 
        img1 = frame_list[t]
        img2 = frame_list[t+1]
        
        ### resize image
        size_multiplier = 64
        H_orig = img1.shape[0]
        W_orig = img1.shape[1]

        H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
        W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)
        
        img1 = cv2.resize(img1, (W_sc, H_sc))
        img2 = cv2.resize(img2, (W_sc, H_sc))
    
        with torch.no_grad():

            ### convert to tensor
            img1 = utils.img2tensor(img1).to(device)
            img2 = utils.img2tensor(img2).to(device)
    
            ### compute fw flow
            fw_flow = model(img1, img2)
            fw_flow = utils.tensor2img(fw_flow)
        
            ### compute bw flow
            bw_flow = model(img2, img1)
            bw_flow = utils.tensor2img(bw_flow)


        ### resize flow
        fw_flow = utils.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig) 
        bw_flow = utils.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig) 
        
        ### compute occlusion
        fw_occ = utils.detect_occlusion(bw_flow, fw_flow)

        noc_mask = 1 - occ_mask


        with torch.no_grad():

            ## convert to tensor
            img1 = utils.img2tensor(img1).to(device)
            flow = utils.img2tensor(flow).to(device)

            ## warp img2
            warp_img1 = flow_warping(img1, flow)

            ## convert to numpy array
            warp_img1 = utils.tensor2img(warp_img1)
        diff = noc_mask*np.abs(img2-warp_img1)

        warping_error += diff.sum()/noc_mask.sum()
    warping_error = warping_error/(len(frame_list)-1)
    return warping_error
        
        