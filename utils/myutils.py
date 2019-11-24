import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error) / X.size(0)
        return loss

class L1_Sobel_Loss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(L1_Sobel_Loss, self).__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(1, 1, 3, bias=False)
        self.conv_op_y = nn.Conv2d(1, 1, 3, bias=False)
        self.conv_op_x = self.conv_op_x.to(device)
        self.conv_op_y = self.conv_op_y.to(device)

        sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, source, target):

        edge_detect_X = torch.ones((source.size(0), 1, source.size(2)-2, source.size(3)-2)).to(self.device)
        edge_detect_Y = torch.ones((source.size(0), 1, source.size(2) - 2, source.size(3) - 2)).to(self.device)

        print(source.size())
        for c in range(3):
            X = source[:, c:c+1, :, :]
            Y = target[:, c:c+1, :, :]
            # print("X_{}_size:{}".format(c, X.size()))
            edge_X_x = self.conv_op_x(X)
            edge_X_y = self.conv_op_y(X)

            edge_Y_x = self.conv_op_x(Y)
            edge_Y_y = self.conv_op_y(Y)

            edge_X = torch.sqrt(edge_X_x * edge_X_x + edge_X_y * edge_X_y)
            edge_Y = torch.sqrt(edge_Y_x * edge_Y_x + edge_Y_y * edge_Y_y)

            edge_detect_X = torch.cat((edge_detect_X, edge_X), dim=1)
            edge_detect_Y = torch.cat((edge_detect_Y, edge_Y), dim=1)

        edge_detect_X = edge_detect_X[:, 1:, :, :]
        edge_detect_Y = edge_detect_Y[:, 1:, :, :]

        diff = torch.add(edge_detect_X, -edge_detect_Y)
        error = torch.sqrt(diff * diff)
        loss = torch.sum(error) / source.size(0)

        return loss

class Weighted_Loss(nn.Module):
    def __init__(self):
        super(Weighted_Loss, self).__init__()
        self.Charbonnier_loss = L1_Charbonnier_loss()
        self.Sobel_Loss = L1_Sobel_Loss(device=torch.device('cuda'))

    def forward(self, X, Y):
        c_loss = self.Charbonnier_loss(X, Y)
        s_loss = self.Sobel_Loss(X, Y)
        loss = c_loss * 0.5 + s_loss * 0.5
        return loss


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach()
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (image_numpy + 1.0) / 2.0
    return image_numpy


def pixel_unshuffle(batch_input, shuffle_scale = 2, device=torch.device('cuda')):
    batch_size = batch_input.shape[0]
    num_channels = batch_input.shape[1]
    height = batch_input.shape[2]
    width = batch_input.shape[3]

    conv1 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv1 = conv1.to(device)
    conv1.weight.data = torch.from_numpy(np.array([[1, 0],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)

    conv2 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv2 = conv2.to(device)
    conv2.weight.data = torch.from_numpy(np.array([[0, 1],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv3 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv3 = conv3.to(device)
    conv3.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [1, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv4 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv4 = conv4.to(device)
    conv4.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [0, 1]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    Unshuffle = torch.ones((batch_size, 4, height//2, width//2), requires_grad=False).to(device)

    for i in range(num_channels):
        each_channel = batch_input[:, i:i+1, :, :]
        first_channel = conv1(each_channel)
        second_channel = conv2(each_channel)
        third_channel = conv3(each_channel)
        fourth_channel = conv4(each_channel)
        result = torch.cat((first_channel, second_channel, third_channel, fourth_channel), dim=1)
        Unshuffle = torch.cat((Unshuffle, result), dim=1)

    Unshuffle = Unshuffle[:, 4:, :, :]
    return Unshuffle.detach()
