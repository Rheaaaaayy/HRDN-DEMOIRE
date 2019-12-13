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
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchnet import meter


from utils.visualize import Visualizer
from utils.myutils import tensor2im, calc_ssim, save_single_image

from models.LossNet import L1_Charbonnier_loss, L1_Sobel_Loss

from data.dataset_val_save_image import Val_MoireData
from models.HRNet import get_pose_net
from models.MSCNN import MSCNN as Sun
from models.DnCNN import DnCNN
from models.Unet import UNet as Unet
from config import cfg


class Config(object):
    is_server = True
    device = torch.device('cuda') if is_server else torch.device('cpu')

    # test_path = "/HDD/sayhi/dataset/TIPbenchmark/test/testData"
    test_path = "T:\\dataset\\moire image benchmark\\test"
    test_batch_size = 100

    num_workers = 6 if is_server else 0

    vis = True if is_server else False
    env = 'test'
    plot_every = 100 #每隔20个batch, visdom画图一次

    model_list = ["HRDN", "DnCNN", "Unet", "Sun"]
    HRDN_model_path = None
    DnCNN_model_path = None
    Unet_model_path = None
    Sun_model_path = None
    save_prefix = "results/"

opt = Config()


def get_model_dict(model_list):
    models = {}
    map_location = lambda storage, loc: storage
    for model_name in model_list:
        if model_name == "HRDN":
            cfg.merge_from_file("config/cfg.yaml")
            model = get_pose_net(cfg, pretrained=opt.model_path)
            model = model.to(opt.device)
            models[model_name] = model
        elif model_name == "DnCNN":
            checkpoints = torch.load(opt.DnCNN_model_path, map_location=map_location)
            model = DnCNN()
            model.load_state_dict(checkpoints["model"])
            model = model.to(opt.device)
            models[model_name] = model
        elif model_name == "Unet":
            checkpoints = torch.load(opt.Unet_model_path, map_location=map_location)
            model = Unet()
            model.load_state_dict(checkpoints["model"])
            model = model.to(opt.device)
            models[model_name] = model
        elif model_name == "Sun":
            checkpoints = torch.load(opt.Sun_model_path, map_location=map_location)
            model = Sun()
            model.load_state_dict(checkpoints["model"])
            model = model.to(opt.device)
            models[model_name] = model
    return models


def test(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.vis:
        vis = Visualizer(opt.env)

    test_data = Val_MoireData(opt.test_path)
    test_dataloader = DataLoader(test_data,
                                batch_size=opt.test_batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                drop_last=False)

    models = get_model_dict(opt.model_list)

    for model_name, model in models.items():
        print(model_name)
        prefix = "{0}{1}/".format(opt.save_prefix, model_name)
        model.eval()
        torch.cuda.empty_cache()
        # criterion_c = L1_Charbonnier_loss()
        # loss_meter = meter.AverageValueMeter()

        psnr_meter = meter.AverageValueMeter()
        ssim_meter = meter.AverageValueMeter()
        vis.log("~~~~~~~~~~~~~~~~~~start test {}~~~~~~~~~~~~~~~~~~~~~~".format(model_name))
        for ii, (moires, clears, labels) in tqdm(enumerate(test_dataloader)):
            moires = moires.to(opt.device)
            clears = clears.to(opt.device)
            if model_name == "HRDN":
                output_list, _ = model(moires)
                outputs = output_list[0]
            else:
                outputs = model(moires)
            moires = tensor2im(moires)
            outputs = tensor2im(outputs)
            clears = tensor2im(clears)

            psnr = colour.utilities.metric_psnr(outputs, clears)
            psnr_meter.add(psnr)

            ssims = 0
            bs = moires.size(0)
            for jj in range(bs):
                output, clear = outputs[jj], clears[jj]
                label = labels[jj]
                img_path = "{0}{1}_output.png".format(prefix, label)
                save_single_image(output, img_path)

                single_ssim = calc_ssim(output, clear)
                ssims += single_ssim
            ssims /= moires.size(0)
            ssim_meter.add(ssims)

            if opt.vis and vis != None and (ii + 1) % 10 == 0:  # 每个个iter画图一次
                vis.log(">>>>>>>> batch_psnr:{psnr}, batch_ssim:{ssim} <<<<<<<<<<".format(psnr=psnr, ssim=ssims))

        print("average psnr is {}, average ssim is {}".format(psnr_meter.value()[0], ssim_meter.value()[0]))
        vis.log("~~~~~~~~~~~~~~~~~~end test {}~~~~~~~~~~~~~~~~~~~~~~".format(model_name))


if __name__ == '__main__':
    test()