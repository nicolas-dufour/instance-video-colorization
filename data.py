from torch.utils.data import Dataset
import os
import cv2
import torch


class PairImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=False):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.transform = transform
        self.inputs_paths = os.listdir(input_dir)
        if self.inputs_paths != os.listdir(output_dir):
            raise "Input and Output not matching"

    def __len__(self):
        return len(self.inputs_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.inputs_paths[idx]

        input_image = cv2.imread(self.input_dir+f"/{image_name}")
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        output_image = cv2.imread(self.output_dir+f"/{image_name}")
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        h, w, _ = input_image.shape
        h, w = (w//32)*32, (h//32)*32
        input_image = input_image[:w,:h,:]
        output_image = output_image[:h,:w,:]
        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)
        return image_name, (input_image, output_image)