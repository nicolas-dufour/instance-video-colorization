import torch
import pytorch_lightning as pl
import torch.nn as nn
from models.unet import UNet
from models.loss import VGGPerceptualLoss
import wandb
import os
from tqdm.notebook import tqdm
import cv2
import ffmpeg as ffmpeg
import glob
import numpy as np
 

class DeepVideoPriorColor(pl.LightningModule):
    def __init__(self, test_loader, loss='perceptual'):
        super().__init__()
        self.test_loader = test_loader
        self.unet = UNet(3, 3, 32)
        initialize_weights(self.unet)
        if loss=='perceptual': 
            self.loss = VGGPerceptualLoss()
        elif loss == 'L1':
            self.loss = nn.L1Loss()
        elif loss == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise "Loss not supported"

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        if(self.current_epoch%10 == 1 and batch_idx==1):
            self.compute_video('output/temp/temp_frames/', 'output/emp/temp_video.mp4')
            wandb.log({"Car": wandb.Video("myvideo.mp4", caption = f"Epoch: {self.current_epoch}")})
            os.remove('output/emp/temp_video.mp4')
            for f in glob.glob("output/temp/temp_frames/*.jpg"):
                os.remove(f)
        _, (grey_image, color_image) = batch
        output = self(grey_image)
        loss = self.loss(output, color_image)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    
    def compute_video(self, output_frames_dir, output_vid_path):
        for _, batch in enumerate(tqdm(self.test_loader)):
            paths = list(batch[0])
            bw_images, _ = batch[1]
            images = model(bw_images.to(self(device)))
            images = 255*torch.clip(images.permute(0,2,3,1),0,1).cpu().detach()
            images = np.uint8(images)
            for i, path in enumerate(paths):
                cv2.imwrite(output_frames_dir+path, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
        ffmpeg.input(f"{output_frames_dir}*.jpg", pattern_type='glob', framerate=25).output(output_vid_dir).run()



def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):        
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()