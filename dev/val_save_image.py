from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import colour

import os
import sys
sys.path.append("/home/publicuser/workspaces/demoire/hrnet-demoire")
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchnet import meter


from utils.visualize import Visualizer
from skimage.metrics import structural_similarity as ssim
from utils.myutils import tensor2im, save_single_image

from models.LossNet import L1_Charbonnier_loss, L1_Sobel_Loss

from data.dataset_val_save_image import Val_MoireData
from models.HRNet import get_pose_net
from ablation import model_fuse, model_finallayer
from models.MSCNN import MSCNN as Sun
from models.DnCNN import DnCNN
from models.Unet import UNet as Unet
from config import cfg


class Config(object):
    is_server = True
    # device = torch.device('cuda') if is_server else torch.device('cpu')
    device = torch.device('cuda')

    test_path = "/HDD/sayhi/dataset/TIPbenchmark/test/testData"
    # test_path = "T:\\dataset\\moire image benchmark\\test"
    test_batch_size = 16

    num_workers = 4 if is_server else 0

    vis = True if is_server else False
    env = 'test'
    plot_every = 10 #每隔20个batch, visdom画图一次

    # model_list = ["HRDN", "DnCNN", "Unet", "Sun"]
    model_list = ["HRDN_finallayer", "HRDN_fuse"]
    HRDN_sobel_path = "checkpoints/ablation/sobel/HRnet_epoch68_1223_07_30_30.pth"
    HRDN_finallayer_path = "checkpoints/ablation/finallayer/HRnet_epoch60_1229_08_09_48.pth"
    HRDN_fuse_path = "checkpoints/ablation/fuse/HRnet_epoch90_1229_11_11_59.pth"
    HRDN_model_path = "checkpoints/TIP_regular_1e-2/HRnet_epoch84_1221_14_43_05.pth"
    DnCNN_model_path = "checkpoints/DnCNN/HRnet_epoch45_1214_11_21_03.pth"
    Unet_model_path = "checkpoints/Unet/HRnet_epoch50_1213_11_32_25.pth"
    Sun_model_path = "checkpoints/MSCNN/HRnet_epoch65_1211_13_58_06.pth"
    save_prefix = "/home/publicuser/workspaces/demoire/all_results/"

opt = Config()

'''
def get_model_dict(model_list):
    models = {}
    map_location = lambda storage, loc: storage
    for model_name in model_list:
        if model_name == "HRDN":
            cfg.merge_from_file("config/cfg.yaml")
            model = get_pose_net(cfg, pretrained=opt.HRDN_model_path)
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
'''

def get_model(model_name):
    map_location = lambda storage, loc: storage
    if model_name == "HRDN_sobel":
        cfg.merge_from_file("config/cfg.yaml")
        model = get_pose_net(cfg, pretrained=opt.HRDN_sobel_path)
        model = model.to(opt.device)
    if model_name == "HRDN_finallayer":
        cfg.merge_from_file("config/cfg.yaml")
        model = model_finallayer.get_pose_net(cfg, pretrained=opt.HRDN_finallayer_path)
        model = model.to(opt.device)
    if model_name == "HRDN_fuse":
        cfg.merge_from_file("config/cfg.yaml")
        model = model_fuse.get_pose_net(cfg, pretrained=opt.HRDN_fuse_path)
        model = model.to(opt.device)

    if model_name == "HRDN":
        cfg.merge_from_file("config/cfg.yaml")
        model = get_pose_net(cfg, pretrained=opt.HRDN_model_path)
        model = model.to(opt.device)
    elif model_name == "DnCNN":
        checkpoints = torch.load(opt.DnCNN_model_path, map_location=map_location)
        model = DnCNN()
        model.load_state_dict(checkpoints["model"])
        model = model.to(opt.device)
    elif model_name == "Unet":
        checkpoints = torch.load(opt.Unet_model_path, map_location=map_location)
        model = Unet()
        model.load_state_dict(checkpoints["model"])
        model = model.to(opt.device)
    elif model_name == "Sun":
        checkpoints = torch.load(opt.Sun_model_path, map_location=map_location)
        model = Sun()
        model.load_state_dict(checkpoints["model"])
        model = model.to(opt.device)
    return model


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


    for model_name in opt.model_list:
        model = get_model(model_name)
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
            if model_name == "HRDN" or model_name == "HRDN_sobel" or model_name == "HRDN_fuse":
                output_list, _ = model(moires)
                outputs = output_list[0]
            elif model_name == "HRDN_finallayer":
                outputs, _ = model(moires)
            else:
                outputs = model(moires)
            moires = tensor2im(moires)
            outputs = tensor2im(outputs)
            clears = tensor2im(clears)

            psnr = colour.utilities.metric_psnr(outputs, clears)
            psnr_meter.add(psnr)

            ssims = 0
            bs = moires.shape[0]
            for jj in range(bs):
                output, clear = outputs[jj], clears[jj]
                label = labels[jj]
                img_path = "{0}{1}_output.png".format(prefix, label)
                save_single_image(output, img_path)

                print(output.size(), clear.size())
                single_ssim = ssim(output, clear, multichannel=True)
                ssims += single_ssim
            ssims /= bs
            ssim_meter.add(ssims)
            if opt.vis and vis != None and (ii + 1) % 10 == 0:  # 每个个iter画图一次
                vis.log(">>>>>>>> batch_psnr:{psnr}, batch_ssim:{ssim} <<<<<<<<<<".format(psnr=psnr, ssim=ssims))

            torch.cuda.empty_cache()
        print("average psnr is {}, average ssim is {}".format(psnr_meter.value()[0], ssim_meter.value()[0]))
        vis.log("~~~~~~~~~~~~~~~~~~end test {}~~~~~~~~~~~~~~~~~~~~~~".format(model_name))


if __name__ == '__main__':
    test()