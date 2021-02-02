import torch
import pytorch_lightning as pl
import torch.nn as nn
from models.unet import UNet
from models.loss import VGGPerceptualLoss


class DeepVideoPriorColor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet(3, 3, 32)
        initialize_weights(self.unet)
        self.loss = VGGPerceptualLoss()
    def forward(self, x):
        return self.unet(x)
    def training_step(self, batch, batch_idx):
        _, (grey_image, color_image) = batch
        output = self.unet(grey_image)
        loss = self.loss(output, color_image)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)



def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):        
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()