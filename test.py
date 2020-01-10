from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colour

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchnet import meter


from utils.visualize import Visualizer
from utils.myutils import tensor2im, save_single_image

from models.LossNet import L1_Charbonnier_loss, L1_Sobel_Loss

from data.dataset_val_save_image import Val_MoireData
from models.HRNet import get_pose_net
from config import cfg

from mypath import Path


class Config(object):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_path, save_prefix= Path.test_dir()
    test_batch_size = 16

    num_workers = 4

    vis = False
    env = 'test'
    plot_every = 10 #每隔10个batch, visdom画图一次

    HRDN_model_path = Path.model_dir()

opt = Config()

def get_model(model_name):
    map_location = lambda storage, loc: storage
    if model_name == "HRDN":
        cfg.merge_from_file("config/cfg.yaml")
        model = get_pose_net(cfg, pretrained=opt.HRDN_model_path)
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


    model = get_model("HRDN")
    prefix = "{0}{1}/".format(opt.save_prefix, "HRDN")
    model.eval()
    torch.cuda.empty_cache()
    # criterion_c = L1_Charbonnier_loss()
    # loss_meter = meter.AverageValueMeter()

    psnr_meter = meter.AverageValueMeter()
    for ii, (moires, clears, labels) in tqdm(enumerate(test_dataloader)):
        moires = moires.to(opt.device)
        clears = clears.to(opt.device)
        output_list, _ = model(moires)
        outputs = output_list[0]
        moires = tensor2im(moires)
        outputs = tensor2im(outputs)
        clears = tensor2im(clears)

        psnr = colour.utilities.metric_psnr(outputs, clears)
        psnr_meter.add(psnr)

        bs = moires.shape[0]
        for jj in range(bs):
            output, clear = outputs[jj], clears[jj]
            label = labels[jj]
            img_path = "{0}{1}_output.png".format(prefix, label)
            save_single_image(output, img_path)

        if opt.vis and vis != None and (ii + 1) % 10 == 0:  # 每10个iter画图一次
            vis.log(">>>>>>>> batch_psnr:{psnr}<<<<<<<<<<".format(psnr=psnr))

        torch.cuda.empty_cache()
    print("average psnr is {}".format(psnr_meter.value()[0]))

if __name__ == '__main__':
    test()