from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict


class MSCNN(nn.Module):
    def __init__(self):
        super(MSCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsamples, self.branches = self.make_branches()
        self.upsamples = self.make_upsamples()
        self.tanh = nn.Tanh()


    def make_one_downsample(self, in_C, inter_C, out_C, is_first=False):
        if is_first == True:
            downsample = nn.Sequential(
                nn.Conv2d(in_C, inter_C, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_C, out_C, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_C, inter_C, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_C, out_C, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
        return downsample


    def make_one_brach(self, inplanes, outplanes, nConv):
        layers = []

        layers.append(nn.Conv2d(inplanes, outplanes, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))
        for i in range(nConv - 1):
            layers.extend([
                nn.Conv2d(outplanes, outplanes, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])

        return nn.Sequential(*layers)

    def make_branches(self):

        # self.branch1 = nn.Sequential(OrderedDict([
        #     ("branch1_down", self.make_one_downsample(3, 32, 32)),
        #     ("branch1", self.make_one_brach(32, 64, 5))
        # ]))

        branch1_down = self.make_one_downsample(3, 32, 32, is_first=True)
        branch1 = self.make_one_brach(32, 64, 5)

        branch2_down = self.make_one_downsample(32, 32, 64)
        branch2 = self.make_one_brach(64, 64, 5)

        branch3_down = self.make_one_downsample(64, 64, 64)
        branch3 = self.make_one_brach(64, 64, 5)

        branch4_down = self.make_one_downsample(64, 64, 64)
        branch4 = self.make_one_brach(64, 64, 5)

        branch5_down = self.make_one_downsample(64, 64, 64)
        branch5 = self.make_one_brach(64, 64, 5)


        downsamples = nn.ModuleList([branch1_down, branch2_down, branch3_down, branch4_down, branch5_down])
        branches = nn.ModuleList([branch1, branch2, branch3, branch4, branch5])

        return downsamples, branches


    def make_one_upsample(self, num_branches):
        if num_branches == 1:
            return nn.Sequential(
                nn.Conv2d(64, 3, 3, 1, 1)
            )
        elif num_branches == 2:
            return nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1)
            )
        elif num_branches == 3:
            return nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1)
            )
        elif num_branches == 4:
            return nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1)
            )
        elif num_branches == 5:
            return nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1)
            )

    def make_upsamples(self):
        upsample_1 = self.make_one_upsample(1)
        upsample_2 = self.make_one_upsample(2)
        upsample_3 = self.make_one_upsample(3)
        upsample_4 = self.make_one_upsample(4)
        upsample_5 = self.make_one_upsample(5)

        upsamples = nn.ModuleList([upsample_1, upsample_2, upsample_3, upsample_4, upsample_5])
        return upsamples


    def forward(self, x):
        upsample_outputs = []
        for i in range(5):
            down_x = self.downsamples[i](x)
            branch_x = self.branches[i](down_x)
            upsample_outputs.append(self.upsamples[i](branch_x))

            x = down_x

        output = upsample_outputs[0]
        for each_upsample in upsample_outputs[1:]:
            output += each_upsample

        return self.tanh(output)