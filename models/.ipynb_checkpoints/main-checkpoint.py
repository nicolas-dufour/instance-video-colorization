import torch
import pytorch_lightning as pl
from models.unet import UNet
from models.loss import VGGPerceptualLoss


class DeepVideoPriorColor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet(3, 3, 32)
        self.loss = VGGPerceptualLoss()

    def training_step(self, batch, batch_idx):
        _, (grey_image, color_image) = batch
        output = self.unet(grey_image)
        loss = self.loss(output, color_image)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)