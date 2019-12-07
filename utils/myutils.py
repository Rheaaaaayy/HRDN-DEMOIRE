import os
import numpy as np
import torch
import torch.nn as nn
import colour
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable


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


def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    return region


def calculate_pasnr(src_path, dst_path):
    src_image_name = os.listdir(src_path)
    dst_image_name = os.listdir(dst_path)
    image_label = ['_'.join(i.split("_")[:-1]) for i in src_image_name]
    num_image = len(src_image_name)
    psnr = 0
    for ii, label in tqdm(enumerate(image_label)):
        src = os.path.join(src_path, "{}_source.png".format(label))
        dst = os.path.join(dst_path, "{}_target.png".format(label))
        src_image = default_loader(src)
        dst_image = default_loader(dst)

        single_psnr = colour.utilities.metric_psnr(src_image, dst_image, 255)
        psnr += single_psnr

    psnr /= num_image
    return psnr


if __name__ == '__main__':
    src_path = "T:\\dataset\\moire image benchmark\\test\\thin_source"
    dst_path = "T:\\dataset\\moire image benchmark\\test\\thin_target"
    psnr = calculate_pasnr(src_path, dst_path)
    print(psnr)