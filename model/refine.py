import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
cu = 32
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, cu)
        self.conv2 = Conv2(cu, 2*cu)
        self.conv3 = Conv2(2*cu, 4*cu)
        self.conv4 = Conv2(4*cu, 8*cu)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]

class downContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputContext0 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding =1)
        self.inputContext1 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding =1)
        self.inputContextfusion = nn.Conv2d(8+3, 16, 3, 1, 1)

    def forward(self, context, origin_image):
        context_d = self.inputContext0(context)
        context_dd = self.inputContext1(context_d)
        out = self.inputContextfusion(torch.cat([origin_image, context_dd], dim = 1))
        return out 
    # out: 16




class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.downforwardori = downContext()
        self.downbackwardori = downContext()
        self.down0 = Conv2(17 + 16+16, 2*cu)
        self.down1 = Conv2(4*cu, 4*cu)
        self.down2 = Conv2(8*cu, 8*cu)
        self.down3 = Conv2(16*cu, 16*cu)
        self.up0 = deconv(32*cu, 8*cu)
        self.up1 = deconv(16*cu, 4*cu)
        self.up2 = deconv(8*cu, 2*cu)
        self.up3 = deconv(4*cu, cu)
        self.conv = nn.Conv2d(cu, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, forwardContext, backwardContext,mask, flow, c0, c1):
        cimg0 = self.downforwardori(forwardContext, img0)
        cimg1 = self.downbackwardori(backwardContext, img1)

        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, cimg0, cimg1,mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.tanh(x)
