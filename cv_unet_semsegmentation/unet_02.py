
"""
Created on Tue Mar 26 17:47:03 2019
@author: Navid
"""
import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.cnn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.cnn8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.cnn9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.cnn10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv1 = nn.Conv2d(1024, 512, kernel_size=1)

        self.cnn11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.cnn12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv2 = nn.Conv2d(512, 256, kernel_size=1)

        self.cnn13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.cnn14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv3 = nn.Conv2d(256, 128, kernel_size=1)

        self.cnn15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv4 = nn.Conv2d(128, 64, kernel_size=1)

        self.cnn17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn19 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c1 = F.relu(self.cnn1(x))
        c2 = F.relu(self.cnn2(c1))
        m1 = self.maxpool1(c2)
        c3 = F.relu(self.cnn3(m1))
        c4 = F.relu(self.cnn4(c3))
        m2 = self.maxpool2(c4)
        c5 = F.relu(self.cnn5(m2))
        c6 = F.relu(self.cnn6(c5))
        m3 = self.maxpool3(c6)
        c7 = F.relu(self.cnn7(m3))
        c8 = F.relu(self.cnn8(c7))
        m4 = self.maxpool4(c8)
        c9 = F.relu(self.cnn9(m4))
        c10 = F.relu(self.cnn10(c9))
        u1 = self.upsample1(c10)
        uc1 = self.upsampleConv1(u1)
        mar_size = int((c8[0].size(1) - uc1[0].size(1)) / 2)
        mar_size2 = int(c8[0].size(1) - mar_size)
        #        import pdb;pdb.set_trace()
        c11 = F.relu(self.cnn11(torch.cat((uc1, c8[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c12 = F.relu(self.cnn12(c11))
        u2 = self.upsample2(c12)
        uc2 = self.upsampleConv2(u2)
        mar_size = int((c6[0].size(1) - uc2[0].size(1)) / 2)
        mar_size2 = int(c6[0].size(1) - mar_size)
        c13 = F.relu(self.cnn13(torch.cat((uc2, c6[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c14 = F.relu(self.cnn14(c13))
        u3 = self.upsample3(c14)
        uc3 = self.upsampleConv3(u3)
        mar_size = int((c4[0].size(1) - uc3[0].size(1)) / 2)
        mar_size2 = int(c4[0].size(1) - mar_size)
        c15 = F.relu(self.cnn15(torch.cat((uc3, c4[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c16 = F.relu(self.cnn16(c15))
        u4 = self.upsample4(c16)
        uc4 = self.upsampleConv4(u4)
        mar_size = int((c2[0].size(1) - uc4[0].size(1)) / 2)
        mar_size2 = int(c2[0].size(1) - mar_size)
        c17 = F.relu(self.cnn17(torch.cat((uc4, c2[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c18 = F.relu(self.cnn18(c17))
        c19 = self.cnn19(c18)
        #        print(out.size())
        #        out = out.view(out.size(0),-1)

        return c19


class UNetModel2(nn.Module):
    def __init__(self):
        super(UNetModel2, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0)
        #        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)
        #        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0)
        #        self.cnn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=0)
        #        self.cnn8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.cnn9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=0)
        #        self.cnn10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv1 = nn.Conv2d(1024, 512, kernel_size=1)

        self.cnn11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=0)
        #        self.cnn12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv2 = nn.Conv2d(512, 256, kernel_size=1)

        self.cnn13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=0)
        #        self.cnn14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv3 = nn.Conv2d(256, 128, kernel_size=1)

        self.cnn15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=0)
        #        self.cnn16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv4 = nn.Conv2d(128, 64, kernel_size=1)

        self.cnn17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0)
        #        self.cnn18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn19 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c1 = F.relu(self.cnn1(x))
        #        c2 = self.cnn2(c1)
        m1 = self.maxpool1(c1)
        c3 = F.relu(self.cnn3(m1))
        #        c4 = self.cnn4(c3)
        m2 = self.maxpool2(c3)
        c5 = F.relu(self.cnn5(m2))
        #        c6 = self.cnn6(c5)
        m3 = self.maxpool3(c5)
        c7 = F.relu(self.cnn7(m3))
        #        c8 = self.cnn8(c7)
        m4 = self.maxpool4(c7)
        c9 = F.relu(self.cnn9(m4))
        #        c10 = self.cnn10(c9)
        u1 = self.upsample1(c9)
        uc1 = self.upsampleConv1(u1)
        mar_size = int((c7[0].size(1) - uc1[0].size(1)) / 2)
        mar_size2 = int(c7[0].size(1) - mar_size)
        #        import pdb;pdb.set_trace()
        c11 = F.relu(self.cnn11(torch.cat((uc1, c7[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        #        c12 = self.cnn12(c11)
        u2 = self.upsample2(c11)
        uc2 = self.upsampleConv2(u2)
        mar_size = int((c5[0].size(1) - uc2[0].size(1)) / 2)
        mar_size2 = int(c5[0].size(1) - mar_size)
        c13 = F.relu(self.cnn13(torch.cat((uc2, c5[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        #        c14 = self.cnn14(c13)
        u3 = self.upsample3(c13)
        uc3 = self.upsampleConv3(u3)
        mar_size = int((c3[0].size(1) - uc3[0].size(1)) / 2)
        mar_size2 = int(c3[0].size(1) - mar_size)
        c15 = F.relu(self.cnn15(torch.cat((uc3, c3[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        #        c16 = self.cnn16(c15)
        u4 = self.upsample4(c15)
        uc4 = self.upsampleConv4(u4)
        mar_size = int((c1[0].size(1) - uc4[0].size(1)) / 2)
        mar_size2 = int(c1[0].size(1) - mar_size)
        c17 = F.relu(self.cnn17(torch.cat((uc4, c1[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        #        c18 = self.cnn18(c17)
        c19 = self.cnn19(c17)
        #        print(out.size())
        #        out = out.view(out.size(0),-1)

        return c19


class UNetModel4(nn.Module):
    def __init__(self):
        super(UNetModel4, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        #        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0)
        #        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.cnn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0)
        #        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.cnn8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0)
        #        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.cnn9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.cnn10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv1 = nn.Conv2d(1024, 512, kernel_size=1)

        self.cnn11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.cnn12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv2 = nn.Conv2d(512, 256, kernel_size=1)

        self.cnn13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.cnn14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv3 = nn.Conv2d(256, 128, kernel_size=1)

        self.cnn15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv4 = nn.Conv2d(128, 64, kernel_size=1)

        self.cnn17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn19 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c1 = F.relu(self.cnn1(x))
        c2 = F.relu(self.cnn2(c1))
        #        m1 = self.maxpool1(c2)
        c3 = F.relu(self.cnn3(c2))
        c4 = F.relu(self.cnn4(c3))
        #        m2 = self.maxpool2(c4)
        c5 = F.relu(self.cnn5(c4))
        c6 = F.relu(self.cnn6(c5))
        #        m3 = self.maxpool3(c6)
        c7 = F.relu(self.cnn7(c6))
        c8 = F.relu(self.cnn8(c7))
        #        m4 = self.maxpool4(c8)
        c9 = F.relu(self.cnn9(c8))
        c10 = F.relu(self.cnn10(c9))
        u1 = self.upsample1(c10)
        uc1 = self.upsampleConv1(u1)
        mar_size = int((c7[0].size(1) - uc1[0].size(1)) / 2)
        mar_size2 = int(c7[0].size(1) - mar_size)
        #        import pdb;pdb.set_trace()
        c11 = F.relu(self.cnn11(torch.cat((uc1, c7[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c12 = F.relu(self.cnn12(c11))
        u2 = self.upsample2(c12)
        uc2 = self.upsampleConv2(u2)
        mar_size = int((c5[0].size(1) - uc2[0].size(1)) / 2)
        mar_size2 = int(c5[0].size(1) - mar_size)
        c13 = F.relu(self.cnn13(torch.cat((uc2, c5[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c14 = F.relu(self.cnn14(c13))
        u3 = self.upsample3(c14)
        uc3 = self.upsampleConv3(u3)
        mar_size = int((c3[0].size(1) - uc3[0].size(1)) / 2)
        mar_size2 = int(c3[0].size(1) - mar_size)
        c15 = F.relu(self.cnn15(torch.cat((uc3, c3[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c16 = F.relu(self.cnn16(c15))
        u4 = self.upsample4(c16)
        uc4 = self.upsampleConv4(u4)
        mar_size = int((c1[0].size(1) - uc4[0].size(1)) / 2)
        mar_size2 = int(c1[0].size(1) - mar_size)
        c17 = F.relu(self.cnn17(torch.cat((uc4, c1[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c18 = F.relu(self.cnn18(c17))
        c19 = self.cnn19(c18)
        #        print(out.size())
        #        out = out.view(out.size(0),-1)

        return c19


class UNetModel5(nn.Module):
    def __init__(self):
        super(UNetModel5, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.cnn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.bn6 = nn.BatchNorm2d(256)

        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.cnn8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.bn8 = nn.BatchNorm2d(512)

        self.cnn9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.cnn10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv1 = nn.Conv2d(1024, 512, kernel_size=1)

        self.cnn11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.cnn12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv2 = nn.Conv2d(512, 256, kernel_size=1)

        self.cnn13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.cnn14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv3 = nn.Conv2d(256, 128, kernel_size=1)

        self.cnn15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.cnn16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampleConv4 = nn.Conv2d(128, 64, kernel_size=1)

        self.cnn17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.cnn19 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c1 = F.relu(self.cnn1(x))
        c2 = F.relu(self.bn2(self.cnn2(c1)))
        m1 = self.maxpool1(c2)
        c3 = F.relu(self.cnn3(m1))
        c4 = F.relu(self.bn4(self.cnn4(c3)))
        m2 = self.maxpool2(c4)
        c5 = F.relu(self.cnn5(m2))
        c6 = F.relu(self.bn6(self.cnn6(c5)))
        m3 = self.maxpool3(c6)
        c7 = F.relu(self.cnn7(m3))
        c8 = F.relu(self.bn8(self.cnn8(c7)))
        m4 = self.maxpool4(c8)
        c9 = F.relu(self.cnn9(m4))
        c10 = F.relu(self.cnn10(c9))
        u1 = self.upsample1(c10)
        uc1 = self.upsampleConv1(u1)
        mar_size = int((c8[0].size(1) - uc1[0].size(1)) / 2)
        mar_size2 = int(c8[0].size(1) - mar_size)
        #        import pdb;pdb.set_trace()
        c11 = F.relu(self.cnn11(torch.cat((uc1, c8[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c12 = F.relu(self.cnn12(c11))
        u2 = self.upsample2(c12)
        uc2 = self.upsampleConv2(u2)
        mar_size = int((c6[0].size(1) - uc2[0].size(1)) / 2)
        mar_size2 = int(c6[0].size(1) - mar_size)
        c13 = F.relu(self.cnn13(torch.cat((uc2, c6[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c14 = F.relu(self.cnn14(c13))
        u3 = self.upsample3(c14)
        uc3 = self.upsampleConv3(u3)
        mar_size = int((c4[0].size(1) - uc3[0].size(1)) / 2)
        mar_size2 = int(c4[0].size(1) - mar_size)
        c15 = F.relu(self.cnn15(torch.cat((uc3, c4[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c16 = F.relu(self.cnn16(c15))
        u4 = self.upsample4(c16)
        uc4 = self.upsampleConv4(u4)
        mar_size = int((c2[0].size(1) - uc4[0].size(1)) / 2)
        mar_size2 = int(c2[0].size(1) - mar_size)
        c17 = F.relu(self.cnn17(torch.cat((uc4, c2[:, :, mar_size:mar_size2, mar_size:mar_size2]), dim=1)))
        c18 = F.relu(self.cnn18(c17))
        c19 = self.cnn19(c18)
        #        print(out.size())
        #        out = out.view(out.size(0),-1)

        return c19

