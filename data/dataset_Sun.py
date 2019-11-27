import os
import os.path
import sys
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


# Crop input as Sun et al.
def default_loader(path):
  img = Image.open(path).convert('RGB')
  w, h = img.size
  region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
  return region


def random_scale_for_pair(moire, clear, is_val=False):
    if is_val == False:
        is_global = np.random.randint(0, 2)
        if is_global == 0:
            resize = transforms.Resize((256, 256))
            moire, clear = resize(moire), resize(clear)
        else:
            resize = transforms.Resize((286, 286))
            moire, clear = resize(moire), resize(clear)

            random_x = np.random.randint(0, moire.size[0] - 256)
            random_y = np.random.randint(0, moire.size[1] - 256)
            moire = moire.crop((random_x, random_y, random_x + 256, random_y + 256))
            clear = clear.crop((random_x, random_y, random_x + 256, random_y + 256))

        is_flip = np.random.randint(0, 2)
        if is_flip == 0:
            moire = moire.transpose(Image.FLIP_LEFT_RIGHT)
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            pass
    else:
        resize = transforms.Resize((256, 256))
        moire, clear = resize(moire), resize(clear)

    return moire, clear

class MoireData(data.Dataset):

    def __init__(self, root, transform=None, is_val=False, loader=default_loader):
        moire_data_root = os.path.join(root, "source")
        clear_data_root = os.path.join(root, "target")

        image_names = os.listdir(clear_data_root)
        image_names = ["_".join(i.split("_")[:-1]) for i in image_names]

        self.moire_images = [os.path.join(moire_data_root, x+"_source.png") for x in image_names]
        self.clear_images = [os.path.join(clear_data_root, x+"_target.png") for x in image_names]

        if transform is not None:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.is_val = is_val
        self.loader = loader

    def downsample(self, clear):
        resize1 = transforms.Resize((128, 128))
        resize2 = transforms.Resize((64, 64))
        resize3 = transforms.Resize((32, 32))

        clear_1 = resize1(clear)
        clear_2 = resize2(clear)
        clear_3 = resize3(clear)

        clear_list = [clear, clear_1, clear_2, clear_3]

        return clear_list

    def __getitem__(self, index):

        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]

        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)

        moire, clear = random_scale_for_pair(moire, clear, self.is_val)

        moire = self.transforms(moire)

        if self.is_val == False:
            clear_list = self.downsample(clear)
            clear_list = [self.transforms(x) for x in clear_list]
            return moire, clear_list
        else:
            clear = self.transforms(clear)
            return moire, clear


    def __len__(self):
        return len(self.moire_images)
