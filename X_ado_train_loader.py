import os
import numpy as np
from PIL import Image
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 



def crop(img, h, w):
    img_np = img.permute(1, 2, 0)  # Convert PIL Image to numpy array
    ih, iw, _ = img_np.shape
    x = random.randint(0, ih - h + 1)
    y = random.randint(0, iw - w + 1)
    img_cropped = img_np[x:x+h, y:y+w, :]
    return img_cropped # Convert back to PIL Image

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to tensor after all other transformations
])

class X4K1000FPSDataset(Dataset):
    def __init__(self, root_dir, transform=transform, crop_size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.image_groups = []
        

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                images = sorted(os.listdir(subdir_path))
                for i in range(0, len(images) - 2):
                    self.image_groups.append([os.path.join(subdir_path, images[i]),
                                              os.path.join(subdir_path, images[i+1]),
                                              os.path.join(subdir_path, images[i+2])])

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        image_paths = self.image_groups[idx]
        timestep = 0.5
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
        if self.crop_size is not None:
            h, w = self.crop_size
        images = [self.transform(image) for image in images]
        img0 = crop(images[0],h,w)
        gt = crop(images[1],h,w)
        img1 = crop(images[2],h,w)
        
        if random.uniform(0, 1) < 0.5:
            img0 = torch.flip(img0, [2])
            img1 = torch.flip(img1, [2])
            gt = torch.flip(gt, [2])

        if random.uniform(0, 1) < 0.5:
            img0 = torch.flip(img0, [1])
            img1 = torch.flip(img1, [1])
            gt = torch.flip(gt, [1])

        if random.uniform(0, 1) < 0.5:
            img0 = torch.flip(img0, [1])
            img1 = torch.flip(img1, [1])
            gt = torch.flip(gt, [1])

        if random.uniform(0, 1) < 0.5:
            img0, img1 = img1, img0  # Swap images
            timestep = 1 - timestep

        if random.uniform(0, 1) < 0.5:
            img0 = torch.flip(img0, [2])  # Flip along the last dimension (width)
            img1 = torch.flip(img1, [2])
            gt = torch.flip(gt, [2])

        img0 = img0.permute(2, 0, 1)
        img1 = img1.permute(2, 0, 1)
        gt = gt.permute(2, 0, 1)
        timestep = torch.full_like(img0, timestep)
        return torch.cat((img0, img1, gt), 0), timestep
        # return image_stack

# Create the dataset
dataset = X4K1000FPSDataset(root_dir='/home/jyzhao/Code/Datasets/X4K1000FPS', transform=transform, crop_size=(224, 224))

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
