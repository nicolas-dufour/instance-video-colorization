import torch
import pytorch_lightning as pl
import torch.nn as nn
from models.unet import UNet
from models.loss import VGGPerceptualLoss
import wandb

class DeepVideoPriorColor(pl.LightningModule):
    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader
        self.unet = UNet(3, 3, 32)
        initialize_weights(self.unet)
        self.loss = VGGPerceptualLoss()

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        if(self.current_epoch%10 == 0):
            self.compute_video('temp/temp_frames/', 'temp/temp_video.mp4')
            wandb.log({"Car": wandb.Video("myvideo.mp4")})
        _, (grey_image, color_image) = batch
        output = self(grey_image)
        loss = self.loss(output, color_image)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    
    def compute_video(self, output_frames_dir, output_vid_path):
        # device = 'cuda:0'
        # model = model.to(device)
        for _, batch in enumerate(tqdm(self.test_loader)):
            paths = list(batch[0])
            bw_images, _ = batch[1]
            images = 255*torch.clip(model(self(device)).permute(0,2,3,1),0,1).cpu().detach()
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