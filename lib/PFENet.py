import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import pvt_v2_b2
import math


class basicconv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(basicconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, 1, 1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class dilatedconv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(dilatedconv, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel // 2, kernel_size=3, dilation=3, padding=3, stride=1)
        self.conv2 = nn.Conv2d(inchannel, outchannel // 2, kernel_size=3, dilation=5, padding=5, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), 1)
        return x


class CMGModule(nn.Module):
    def __init__(self):
        super(CMGModule, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, 1)
        self.conv2 = nn.Conv2d(320, 128, 1)
        self.conv3 = nn.Conv2d(128, 64, 1)
        self.conv4 = nn.Conv2d(64, 48, 1)
        self.upconv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.upconv2 = nn.Conv2d(384, 384, 3, 1, 1)
        self.upconv3 = nn.Conv2d(448, 448, 3, 1, 1)

        self.choose = nn.Conv2d(496, 1, 1, bias='False')

    def forward(self, x1, x2, x3, x4):  # x1smallest x4biggest
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x1 = self.upconv1(x1)
        x2 = torch.cat((x1, x2), dim=1)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x2 = self.upconv2(x2)
        x3 = torch.cat((x2, x3), dim=1)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x3 = self.upconv3(x3)
        x4 = torch.cat((x3, x4), dim=1)
        sg = self.choose(x4)

        return sg


class CFEModule(nn.Module):
    def __init__(self, c_low, c_x, c_high, outc):
        super(CFEModule, self).__init__()

        self.conv_x = nn.Sequential(
            dilatedconv(c_x, c_x),
            nn.Conv2d(c_x, outc, 1))

        self.conv_low = nn.Sequential(
            nn.Conv2d(c_low, c_low, 3, 1, 1),
            nn.Conv2d(c_low, outc, 1)
        )

        self.conv_high = nn.Sequential(
            nn.Conv2d(c_high, c_high, 3, 1, 1),
            nn.Conv2d(c_high, outc, 1)
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(c_high, outc)

        self.convsa = nn.Conv2d(2, 1, 7, 1, 3)

        self.output = nn.Sequential(
            nn.Conv2d(outc, outc, 1)
        )

        self.chigh = c_high
        self.cout = outc

    def forward(self, low, x, high):
        # resize
        B, _, H, W = x.size()
        low = F.interpolate(low, size=(H, W), mode='bilinear')
        high = F.interpolate(high, size=(H, W), mode='bilinear')

        # convolution
        x = self.conv_x(x)
        high_res = self.conv_high(high)

        low_res = self.conv_low(low)

        # channel attention
        ca = self.GAP(high).view(B, self.chigh)
        ca = self.linear(ca)
        ca = self.sigmoid(ca).view(B, self.cout, 1, 1)

        # spatial attention
        avglow = torch.mean(low, dim=1, keepdim=True)
        maxlow, _ = torch.max(low, dim=1, keepdim=True)
        sa = torch.cat((avglow, maxlow), dim=1)
        sa = self.convsa(sa)
        sa = self.sigmoid(sa)

        # output
        x_low = low_res * sa
        x_high = high_res * ca
        result = self.output(x + x_low + x_high)

        return result


class FBCModule(nn.Module):
    def __init__(self, inchannel):
        super(FBCModule, self).__init__()
        self.edge1 = basicconv(inchannel, inchannel)
        self.edge2 = basicconv(inchannel, inchannel)
        self.foreground = basicconv(inchannel, inchannel)
        self.fuse = basicconv(inchannel * 2, 64)
        self.convfeat = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # self.at = _NonLocalBlockND(64, None, 2)
        self.conv = nn.Conv2d(64, 64, 3, padding=1)
        self.output = nn.Conv2d(64, 1, 1, bias=False)

    def forward(self, x, priorfeat, priormap, sg):
        # resize
        _, _, H, W = x.size()
        priormap = F.interpolate(priormap, size=(H, W), mode='bilinear')
        sg = F.interpolate(sg, size=(H, W), mode='bilinear')
        priorfeat = F.interpolate(priorfeat, size=(H, W), mode='bilinear')

        # edge
        x_1 = self.edge1(x)
        x_2 = self.edge2(x)
        x_1 = x_1 * sg
        x_2 = x_2 * priormap
        edge = x_1 - x_2

        # foreground
        x_f = self.foreground(x)
        foreground = x_f * sg

        # edge and foreground
        add = torch.cat((edge, foreground), dim=1)
        add = self.fuse(add)

        # feat
        priorfeat = self.convfeat(priorfeat) + add
        outfeat = self.conv(priorfeat)
        outmap = self.output(outfeat)

        return outfeat, outmap


class PFENet(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        # ---- PVT Backbone ----
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './lib/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.decoder = CMGModule()

        self.CBR = basicconv(512, 64)
        self.mcff22 = CFEModule(128, 320, 512, 64)
        self.mcff44 = CFEModule(64, 128, 320, 64)
        self.mcff88 = CFEModule(64, 64, 128, 64)

        self.firstmap = nn.Conv2d(64, 1, 1, bias=False)

        self.f11 = FBCModule(64)
        self.f22 = FBCModule(64)
        self.f44 = FBCModule(64)
        self.f88 = FBCModule(64)

    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]  # bs,64,88,88
        x2 = pvt[1]  # bs,128,44,44
        x3 = pvt[2]  # bs,320,22,22
        x4 = pvt[3]  # bs,512,11,11

        coarse_map = self.decoder(x4, x3, x2, x1)

        y4 = self.CBR(x4)
        y3 = self.mcff22(x2, x3, x4)
        y2 = self.mcff44(x1, x2, x3)
        y1 = self.mcff88(x1, x1, x2)

        firstmap = self.firstmap(y4)

        feat11, map11 = self.f11(y4, y4, firstmap, coarse_map)
        feat22, map22 = self.f22(y3, feat11, map11, coarse_map)
        feat44, map44 = self.f44(y2, feat22, map22, coarse_map)
        feat88, map88 = self.f88(y1, feat44, map44, coarse_map)

        lateral_map_1 = F.interpolate(map88, scale_factor=4, mode='bilinear')
        lateral_map_2 = F.interpolate(map44, scale_factor=8, mode='bilinear')
        lateral_map_3 = F.interpolate(map22, scale_factor=16, mode='bilinear')
        lateral_map_4 = F.interpolate(map11, scale_factor=32, mode='bilinear')
        lateral_map_5 = F.interpolate(coarse_map, scale_factor=4, mode='bilinear')

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1
