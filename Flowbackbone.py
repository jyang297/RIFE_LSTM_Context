import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from Myflow import *

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##
# input x: test image from Vim90K


class Model:
    def __init__(self, args):
        if args.flownet == 'Myflow':
            self.flownet = Myflow()
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        self.fnet = None
    # def ResidualOutput(nn.Module):
        # Procedure:
        # Frame 0 + Frame 1 + Hiddenstate ==> Forward Optical Flow  
        # 
        # input size should be the size of h
        # Output dim should be the size of Optical flow, which means 2
        # The two frames are RGB images. So the feature should be 3(6 in total)

    def forward(self, frame0, frame1, frameGT, frame3, frame4, iters=12, flow_init=None, upsample=True, test_mode=False):
        frame0 = 2 * (frame0 / 255.0) - 1.0
        frame1 = 2 * (frame1 / 255.0) - 1.0
        frame3 = 2 * (frame3 / 255.0) - 1.0
        frame4 = 2 * (frame4 / 255.0) - 1.0

        frameGT = 2 * (frameGT / 255.0) - 1.0
        down_h = frame0.size(dim=-1) // 2
        down_w = frame0.size(dim=-2) // 2
# Perform downsampling using interpolate
        downframe0 = F.interpolate(frame0.unsqueeze(0), size=(down_h, down_w), mode='bilinear', align_corners=False)
        downframe1 = F.interpolate(frame0.unsqueeze(0), size=(down_h, down_w), mode='bilinear', align_corners=False)
        hdim = self.hidden_dim
        cdim = self.context_dim

        # Block 1: Extract features from origin images
        with autocast(enabled=self.args.mixed_precision):
            # Plan A: Correlation
            # fmap0, fmap1 = self.fnet(downframe0, downframe1)
            
            # Plan B: Only features
            





    


            



    