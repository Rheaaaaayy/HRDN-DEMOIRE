from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import colour

import fire
import os
import sys
import ipdb
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchnet import meter
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.checkpoint import checkpoint


from utils.visualize import Visualizer


import models
from models.HRNet import get_pose_net
from config import cfg, update_config
# from myconfig import opt
from data.dataset import MoireData

class Config(object):
    temp_winorserver = False
    is_dev = True if temp_winorserver else False
    is_linux = False if temp_winorserver else True
    gpu = False if temp_winorserver else True # 是否使用GPU
    device = torch.device('cuda') if gpu else torch.device('cpu')

    if is_linux == False:
        train_path = "T:\\dataset\\AIM2019 demoireing challenge\\Training\\Training"
        valid_path = "T:\\dataset\\AIM2019 demoireing challenge\\Validation"
    else:
        train_path = "/home/publicuser/sayhi/dataset/demoire/Training"
        valid_path = "/home/publicuser/sayhi/dataset/demoire/Validation"
    label_dict = {1: "moire",
                  0: "clear"}
    num_workers = 4
    image_size = 64
    train_batch_size = 2 #train的维度为(2, 3, 256, 256)
    val_batch_size = 2
    max_epoch = 200
    lr = 0.0002
    beta1 = 0.5  # Adam优化器的beta1参数


    vis = False if temp_winorserver else True
    env = 'demoire'
    plot_every = 40 #每隔20个batch, visdom画图一次

    debug_file = 'F:\\workspaces\\learn_pytorch\\GAN\\debug'  # 存在该文件则进入debug模式

    save_every = 10  # 每10个epoch保存一次模型
    model_path = None #'checkpoints/HRnet_211.pth'

opt = Config()


def train(**kwargs):
    #init
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.vis:
        vis = Visualizer(opt.env)

    #dataset
    # train_transforms = transforms.Compose([
    #     transforms.FiveCrop(256),
    #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    # ])
    data_transforms = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    train_data = MoireData(opt.train_path, data_transforms)
    val_data = MoireData(opt.valid_path, data_transforms)
    train_dataloader = DataLoader(train_data,
                            batch_size=opt.train_batch_size if opt.is_dev == False else 4,
                            shuffle=True,
                            num_workers=opt.num_workers if opt.is_dev == False else 0,
                            drop_last=True)
    val_dataloader = DataLoader(val_data,
                            batch_size=opt.val_batch_size if opt.is_dev == False else 4,
                            shuffle=True,
                            num_workers=opt.num_workers if opt.is_dev == False else 0,
                            drop_last=True)

    #model_init
    cfg.merge_from_file("config/cfg.yaml")
    model = get_pose_net(cfg) #initweight
    map_location = lambda storage, loc: storage
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path, map_location=map_location))

    model = model.to(opt.device)

    criterion = nn.MSELoss(reduction='mean')
    lr = opt.lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0.0001
    )

    loss_meter = meter.AverageValueMeter()
    psnr_meter = meter.AverageValueMeter()
    previous_loss = 1e100
    accumulation_steps = 8

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        psnr_meter.reset()

        for ii, (moires, clears) in tqdm(enumerate(train_dataloader)):
            # bs, ncrops, c, h, w = moires.size()
            moires = moires.to(opt.device)
            clears = clears.to(opt.device)

            outputs = model(moires)
            loss = criterion(outputs, clears)

            psnr = colour.utilities.metric_psnr(outputs.detach().cpu().numpy(), clears.cpu().numpy())
            #saocaozuo gradient accumulation
            loss = loss/accumulation_steps
            loss.backward()

            if (ii+1)%accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_meter.add(loss.item())
            psnr_meter.add(psnr)

            if opt.vis and (ii + 1) % opt.plot_every == 0: #40个batch画图一次
                vis.images(moires.detach().cpu().numpy(), win='moire_image')
                vis.images(outputs.detach().cpu().numpy(), win='output_image')
                vis.images(clears.cpu().numpy(), win='clear_image')

                vis.plot('train_loss', loss_meter.value()[0]) #meter.value() return 2 value of mean and std
                vis.log("epoch:{epoch}, lr:{lr}, train_loss:{loss}, train_psnr:{train_psnr}".format(epoch=epoch,
                                                                                          loss=loss_meter.value()[0],
                                                                                          lr=lr,
                                                                                          train_psnr = psnr_meter.value()[0]))
        val_loss, val_psnr = val(model, val_dataloader)
        vis.plot('val_loss', val_loss)
        vis.log("epoch:{epoch}, lr:{lr}, val_loss:{val_loss}, val_psnr:{val_psnr}".format(epoch=epoch,
                                                                                            loss=val_loss,
                                                                                            lr=lr,
                                                                                            val_psnr=val_psnr))

        if (epoch + 1) % opt.save_every == 0: # 10个epoch保存一次
            prefix = 'checkpoints/HRnet_epoch{}_'.format(epoch+1)
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            torch.save(model.state_dict(), name)

        ''' lr decay
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]
        '''

    prefix = 'checkpoints/HRnet_final_'
    name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
    torch.save(model.state_dict(), name)

'''
    # plot the example
    for moires, clears in dataloader:
        bs, ncrops, c, h, w = moires.size()
        moires = moires.view(-1, c, h, w)
        clears = clears.view(-1, c, h, w)

        for j, (moire, clear) in enumerate(zip(moires, clears)):
            plt.subplot(2, 10, 2 * (j + 1) - 1)
            moire = moire.numpy().transpose(1, 2, 0)
            plt.imshow(moire)
            plt.title(j)

            plt.subplot(2, 10, 2 * (j + 1))
            clear = clear.numpy().transpose(1, 2, 0)
            plt.imshow(clear)
            plt.title(j)

            psnr = colour.utilities.metric_psnr(moire, clear, max_a=1)
            print(psnr)

            if j == 9:
                break
        plt.show()
        break
'''


@torch.no_grad()
def val(model, dataloader):
    model.eval()
    criterion = nn.MSELoss()

    loss_meter = meter.AverageValueMeter()
    psnr_meter = meter.AverageValueMeter()

    for ii, (val_moires, val_clears) in tqdm(enumerate(dataloader)):
        val_moires = val_moires.to(opt.device)
        val_clears = val_clears.to(opt.device)
        val_outputs = model(val_moires)

        loss = criterion(val_outputs, val_clears)
        psnr = colour.utilities.metric_psnr(val_outputs, val_clears)

        loss_meter.add(loss.item())
        psnr_meter.add(psnr)

    model.train()
    return loss_meter.value()[0], psnr_meter.value()[0]



if __name__ == '__main__':
    '''
    dummy_input = torch.rand(
        (1, 3, 256, 256)
    )

    cfg.merge_from_file("config/cfg.yaml")
    model = get_pose_net(cfg)
    model = model.cuda()

    input_ = dummy_input.clone()
    input_ = input_.cuda()

    output = model(input_)
    '''

    train()




    # # final_layer = model.final_layer
    #
    # # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # # print(para)
    # # print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
