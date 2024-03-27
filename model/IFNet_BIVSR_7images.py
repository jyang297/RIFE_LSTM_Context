import torch
import torch.nn as nn
import torch.nn.functional as F
from model.laplacian import *
from model.warplayer import warp
from model.refine import *
from model.myContext import *
from model.loss import *
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
    
class Modified_Simplified_IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=150)
        self.block2 = IFBlock(13+4, c=90)
        self.block_tea = IFBlock(16+4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()


    def forward(self, x, scale=[4,2,1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        

        for i in range(3):
            # merged (list): --> 3*2 --> RGB * (Forward, Backward) 
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        
        # meltpre = self.mycontext(img0, img1, warped_img0, warped_img1, mask, flow)
        #for i in range(3):
        # loss_meltpre = (((meltpre -gt)**2).mean(1,True)** 0.5 * loss_mask).mean()

        # merged[i]: (warped_img0, warped_img1)

        tmp = img0, img1, warped_img0, warped_img1, mask, flow

        pass
        return merged[2], loss_distill

class Modified_IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lap = LapLoss()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=150)
        self.block2 = IFBlock(13+4, c=90)
        self.block_tea = IFBlock(16+4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, forwardContext, backwardContext, scale=[4,2,1]):
        # forwardContext/backwardContext is forwardFeature[i], only pick up the one for the current interpolation
        # final_merged, loss = self.mIFnetframesset[i], for(wardFeature[3*i], backwardFeature[3*i+2])
        
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        eps = 1e-8

        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        # Temporally put timeframeFeatrues here
        # modified to tanh as output ~[-1,1]
        tmp = self.unet(img0, img1, warped_img0, warped_img1, forwardContext, backwardContext,mask, flow, c0, c1)
        # tPredict = tmp[:, :3] * 2 - 1
        # predictimage = tmp
        #merged[2] = torch.clamp(tPredict, 0, 1)
        predictimage = torch.clamp(merged[2] + tmp, 0, 1)
        loss_pred = (((predictimage - gt) **2+eps).mean(1,True)**0.5).mean()
        
        #loss_pred = (((merged[2] - gt) **2).mean(1,True)**0.5).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()

        return flow_list, mask_list[2], predictimage, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred
     
# Change this to attention in the future
class FBwardExtractor(nn.Module):
    # input 3 outoput 6
    def __init__(self, in_plane=3, out_plane=6):
        super().__init__()
        self.fromimage = conv(in_plane, 6, kernel_size=3, stride=1, padding=1)
        self.conv0 = nn.Sequential(
            conv(6, 12, 3, 1, 1),
            conv(12, out_plane, 3, 1, 1),
            )

    def forward(self, x):
        x = self.fromimage(x)
        x = x+ self.conv0(x)
        return x

# Chagne this to GRU in the future 
class fakeLSTMBlock(nn.Module):
    def __init__(self, in_planes = 3+6, out_planes = 6):
        super().__init__()
        self.fLIn = conv(in_planes, 12)
        self.middleconv0 = nn.Sequential(
            conv(12, 24),
            conv(24, 24),
            conv(24, 24),
            conv(24, 24),
            conv(24, 24)
        )
        self.fLOut0 = conv(24, 12)
        self.fLOut1 = conv(12,out_planes=out_planes)

    def forward(self, x):
        # x = previous feature , current_time_image/feature
        y = self.fLIn(x)
        x = self.middleconv0(y)
        x = y + self.fLOut0(x) 
        x = self.fLOut1(x)
        return x

# Reconstruction
class Reconstruction(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x):
        # INput: 3+3+C+C  warped_img0 forwardFeature[i], warped_img1 backwardFeature[i]
        # Output: 3
        
        return 0
      


# Backbone of the 7 images for loop
class VSRbackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fLstm012 = fakeLSTMBlock()
        self.fLstm234 = fakeLSTMBlock()
        self.fLstm456 = fakeLSTMBlock()
        self.bLstm012 = fakeLSTMBlock()
        self.bLstm234 = fakeLSTMBlock()
        self.bLstm456 = fakeLSTMBlock()

        self.FirstExtractor = FBwardExtractor()
        self.LastExtractor = FBwardExtractor()

        
        self.Fusion = Modified_IFNet()
#####
        # merged[i]  (warped_img0, warped_img1)
        # warped_img0 = warp(img0, flow[:, :2])
        # warped_img1 = warp(img1, flow[:, 2:4])

    def forward(self, allframes):

        # using septuplet: Batch*21*224*224
        forwardFeature = []
        backwardFeature = []
        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_distill = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_tea = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        
        # LSTM LOOP
        # input：
        #   Forward:  img0 and img1 --> forward feature at 0.5
        #   Backward: img1 and img0 --> backward feature at 0.5
        # FirstExtractor: input 3 --> output 6    
        img0 = allframes[:,:3]
        img6 = allframes[:,-3:] 
        fimg0 = self.FirstExtractor(img0) # put frame 0 here
        # fLSTM: input: 3+6 output: 6
        forwardFeature.append(self.fLstm012(torch.cat([fimg0, allframes[:,6:9]], dim=1))) # cat image 1 here: frame2
        forwardFeature.append(self.fLstm234(torch.cat([forwardFeature[0], allframes[:,12:15]], dim=1))) # cat image 1 here: frame4
        forwardFeature.append(self.fLstm456(torch.cat([forwardFeature[1], img6], dim=1))) # cat image 1 here: frame6
        # forwardFeature[]: [forward feature for frame1, frame3, frame5]

        # FirstExtractor: input 3 --> output 6    
        fimg6 = self.LastExtractor(img6) # put frame 6 here
        # bLSTM: input: 3+6 output: 6
        # input : previous bLSTM cat new frame
        backwardFeature.append(self.bLstm456(torch.cat([fimg6, allframes[:,12:15]], dim=1))) # cat image 1 here: frame4
        backwardFeature.append(self.bLstm234(torch.cat([backwardFeature[0], allframes[:,6:9]], dim=1))) # cat image 1 here: frame2
        backwardFeature.append(self.bLstm012(torch.cat([backwardFeature[1], img0], dim=1))) # cat image 1 here: frame0
        # backwardFeature[]: [backward feature for frame5, frame3, frame1]
        # return forwardFeature, backwardFeature
        output_allframes = []
        output_onlyteacher = []
        flow_list = []
        flow_teacher_list = []
        mask_list = []
        # Now iterate through septuplet and get three inter frames
        for i in range(3):
            img0 = allframes[:, 6*i:6*i+3]
            gt = allframes[:, 6*i+3:6*i+6]
            img1 = allframes[:, 6*i+6:6*i+9]
            imgs_and_gt = torch.cat([img0,img1,gt],dim=1)

            flow, mask, merged, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred = self.Fusion(imgs_and_gt, forwardFeature[i], backwardFeature[-i-1] )
            # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(allframes)
            Sum_loss_distill += loss_distill 
            Sum_loss_context += 1/3 * loss_pred
            Sum_loss_tea +=loss_tea
            output_allframes.append(img0)
            # output_allframes.append(merged[2])
            output_allframes.append(merged)
            flow_list.append(flow)
            flow_teacher_list.append(flow_teacher)
            output_onlyteacher.append(merged_teacher)
            mask_list.append(mask)

            # The way RIFE compute prediction loss and 
            # loss_l1 = (self.lap(merged[2], gt)).mean()
            # loss_tea = (self.lap(merged_teacher, gt)).mean()

        output_allframes.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        pass


        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_onlyteacher, Sum_loss_distill, Sum_loss_context, Sum_loss_tea
            


        











