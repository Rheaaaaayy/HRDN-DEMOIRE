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
import time
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
from utils.myutils import tensor2im

from models.LossNet import L1_Charbonnier_loss, L1_Sobel_Loss


import models
from models.HRNet_dev import get_pose_net
from models.MSCNN import MSCNN
from config import cfg, update_config
# from myconfig import opt
from data.dataset_Sun import MoireData

class Config(object):
    temp_winorserver = False
    is_dev = True if temp_winorserver else False
    is_linux = False if temp_winorserver else True
    gpu = False if temp_winorserver else True # 是否使用GPU
    device = torch.device('cuda') if gpu else torch.device('cpu')

    if is_linux == False:
        train_path = "T:\\dataset\\moire image benchmark\\train"
        valid_path = "T:\\dataset\\moire image benchmark\\val"
        test_path = "T:\\dataset\\moire image benchmark\\test"
        debug_file = 'F:\\workspaces\\demoire\\debug'  # 存在该文件则进入debug模式
    else:
        train_path = "/seconddisk/sayhi/dataset/TIPbenchmark/train/trainData"
        # valid_path = "/home/publicuser/sayhi/dataset/moire image benchmark/val"
        test_path = "/seconddisk/sayhi/dataset/TIPbenchmark/test/testData"
        debug_file = '/home/publicuser/sayhi/demoire/HRnet-demoire/debug'  # 存在该文件则进入debug模式
    label_dict = {1: "moire",
                  0: "clear"}
    num_workers = 6
    image_size = 256
    train_batch_size = 32 #train的维度为(10, 3, 256, 256) 一个batch10张照片，要1000次iter
    val_batch_size = 32
    max_epoch = 400
    lr = 1e-4
    lr_decay = 0.90
    beta1 = 0.5  # Adam优化器的beta1参数
    accumulation_steps = 1 #梯度累加的参数
    loss_alpha = 0.6 #两个loss的权值

    vis = False if temp_winorserver else True
    env = 'demoire'
    plot_every = 100 #每隔20个batch, visdom画图一次

    save_every = 5  # 每5个epoch保存一次模型
    model_path = None #'checkpoints/HRnet_211.pth'
    save_prefix = "checkpoints/benchmark_multi_loss/"

opt = Config()


def train(**kwargs):
    #init
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.vis:
        vis = Visualizer(opt.env)
        vis_val = Visualizer('valdemoire')

    #dataset
    FiveCrop_transforms = transforms.Compose([
        transforms.FiveCrop(256),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])
    data_transforms = transforms.Compose([
        # transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    train_data = MoireData(opt.train_path)
    test_data = MoireData(opt.test_path, is_val=True)
    train_dataloader = DataLoader(train_data,
                            batch_size=opt.train_batch_size if opt.is_dev == False else 4,
                            shuffle=True,
                            num_workers=opt.num_workers if opt.is_dev == False else 0,
                            drop_last=True)
    test_dataloader = DataLoader(test_data,
                            batch_size=opt.val_batch_size if opt.is_dev == False else 4,
                            shuffle=True,
                            num_workers=opt.num_workers if opt.is_dev == False else 0,
                            drop_last=True)

    last_epoch = 0
    #model_init
    cfg.merge_from_file("config/cfg.yaml")
    model = get_pose_net(cfg, pretrained=opt.model_path) #initweight
    model = model.to(opt.device)

    # val_loss, val_psnr = val(model, test_dataloader, vis_val)
    # print(val_loss, val_psnr)

    criterion_c = L1_Charbonnier_loss()
    criterion_s = L1_Sobel_Loss()
    lr = opt.lr
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        # filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.0001
    )

    if opt.model_path:
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(opt.model_path, map_location=map_location)
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)

        lr = checkpoint["lr"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    loss_meter = meter.AverageValueMeter()
    psnr_meter = meter.AverageValueMeter()
    previous_loss = 1e100
    accumulation_steps = opt.accumulation_steps


    for epoch in range(opt.max_epoch):
        if epoch < last_epoch:
            continue
        loss_meter.reset()
        psnr_meter.reset()
        loss_list = []
        print(epoch)

        for ii, (moires, clear_list) in tqdm(enumerate(train_dataloader)):
            torch.cuda.empty_cache()
            # bs, ncrops, c, h, w = moires.size()
            moires = moires.to(opt.device)
            clears = clear_list[0].to(opt.device)
            clear_list = [x.to(opt.device) for x in clear_list]

            output_list, edge_output_list = model(moires)
            outputs, edge_X = output_list[0], edge_output_list[0]

            loss = 0
            if epoch < 50:
                if ii > 1562:
                    break
                for jj, (output, edge_output) in enumerate(zip(output_list, edge_output_list)):
                    c_loss = criterion_c(output, clear_list[jj])
                    s_loss = criterion_s(edge_output, clear_list[jj])
                    if jj == 0:
                        loss += 0.25 * (c_loss * 0.7 + s_loss * 0.3)
                    elif jj <= 2:
                        loss += 0.25 * (c_loss * 0.75 + s_loss * 0.25)
                    elif jj > 2:
                        loss += 0.25 * (c_loss * 0.9 + s_loss * 0.1)
            else:
                c_loss = criterion_c(outputs, clears)
                s_loss = criterion_s(edge_X, clears)
                loss = (c_loss * opt.loss_alpha + s_loss * opt.loss_alpha)

            # saocaozuo gradient accumulation
            loss = loss/accumulation_steps
            loss.backward()

            if (ii+1)%accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_meter.add(loss.item()*accumulation_steps)

            moires = tensor2im(moires)
            outputs = tensor2im(outputs)
            clears = tensor2im(clears)

            psnr = colour.utilities.metric_psnr(outputs, clears)
            psnr_meter.add(psnr)

            if opt.vis and (ii + 1) % opt.plot_every == 0: #100个batch画图一次
                vis.images(moires, win='moire_image')
                vis.images(outputs, win='output_image')
                vis.text("current outputs_size:{outputs_size},<br/> outputs:{outputs}<br/>".format(
                                                                                    outputs_size=outputs.shape,
                                                                                    outputs=outputs), win="size")
                vis.images(clears, win='clear_image')
                #record the train loss to txt
                vis.plot('train_loss', loss_meter.value()[0]) #meter.value() return 2 value of mean and std
                vis.log("epoch:{epoch}, lr:{lr}, train_loss:{loss}, train_psnr:{train_psnr}".format(epoch=epoch+1,
                                                                                          loss=loss_meter.value()[0],
                                                                                          lr=lr,
                                                                                          train_psnr = psnr_meter.value()[0]))
                loss_list.append(str(loss_meter.value()[0]))
                # if os.path.exists(opt.debug_file):
                #     ipdb.set_trace()

        torch.cuda.empty_cache()
        val_loss, val_psnr = val(model, test_dataloader, vis_val)
        if opt.vis:
            vis.plot('val_loss', val_loss)
            vis.log("epoch:{epoch}, average val_loss:{val_loss}, average val_psnr:{val_psnr}".format(epoch=epoch+1,
                                                                                            val_loss=val_loss,
                                                                                            val_psnr=val_psnr))
        #每个epoch把loss写入文件
        with open(opt.save_prefix+"loss_list.txt", 'a') as f:
            f.write("\nepoch_{}\n".format(epoch+1))
            f.write('\n'.join(loss_list))

        if (epoch + 1) % opt.save_every == 0 or epoch == 0: # 每5个epoch保存一次
            prefix = opt.save_prefix+'HRnet_epoch{}_'.format(epoch+1)
            file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            checkpoint = {
                'epoch': epoch + 1,
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "lr": lr
            }
            torch.save(checkpoint, file_name)

        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


    prefix = opt.save_prefix+'HRnet_final_'
    file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
    checkpoint = {
        'epoch': epoch + 1,
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(),
        "lr": lr
    }
    torch.save(checkpoint, file_name)


@torch.no_grad()
def val(model, dataloader, vis=None):
    model.eval()

    criterion_c = L1_Charbonnier_loss()
    criterion_s = L1_Sobel_Loss()

    loss_meter = meter.AverageValueMeter()
    psnr_meter = meter.AverageValueMeter()
    for ii, (val_moires, val_clears) in tqdm(enumerate(dataloader)):
        val_moires = val_moires.to(opt.device)
        val_clears = val_clears.to(opt.device)
        val_output_list, val_edge_output_list = model(val_moires)
        val_outputs, val_edge_outputs = val_output_list[0], val_edge_output_list[0]

        c_loss = criterion_c(val_outputs, val_clears)
        s_loss = criterion_s(val_edge_outputs, val_clears)
        val_loss = c_loss * opt.loss_alpha + s_loss * (1 - opt.loss_alpha)

        loss_meter.add(val_loss.item())

        val_moires = tensor2im(val_moires)
        val_outputs = tensor2im(val_outputs)
        val_clears = tensor2im(val_clears)

        val_psnr = colour.utilities.metric_psnr(val_outputs, val_clears)
        psnr_meter.add(val_psnr)

        if opt.vis and vis != None and (ii + 1) % 100 == 0:  # 每个个iter画图一次
            vis.images(val_moires, win='val_moire_image')
            vis.images(val_outputs, win='val_output_image')
            vis.images(val_clears, win='val_clear_image')

            vis.log(">>>>>>>> val_loss:{val_loss}, val_psnr:{val_psnr}".format(val_loss=val_loss,
                                                                             val_psnr=val_psnr))

    model.train()
    return loss_meter.value()[0], psnr_meter.value()[0]


if __name__ == '__main__':
    # train(model_path='checkpoints/benchmark_multi_loss/HRnet_epoch1_1127_12_47_17.pth')
    train()