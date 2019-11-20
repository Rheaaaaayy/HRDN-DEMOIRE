import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms

class MoireData(data.Dataset):

    def __init__(self, root, transform=None, is_val=False):
        moire_data_root = os.path.join(root, "moire")
        clear_data_root = os.path.join(root, "clear")

        image_names = os.listdir(clear_data_root)
        self.moire_images = [os.path.join(moire_data_root, x) for x in image_names]
        self.clear_images = [os.path.join(clear_data_root, x) for x in image_names]

        if transform is not None:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.ToTensor()
            ])

        self.is_val = is_val

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]

        moire = Image.open(moire_img_path).convert('RGB')
        clear = Image.open(clear_img_path).convert('RGB')
        if self.is_val == False:
            is_global = np.random.randint(0, 2)
            resize = transforms.Resize(256)
            if is_global==0:
                moire = resize(moire)
                clear = resize(clear)
            else:
                random_x =np.random.randint(0, moire.size[0]-256)
                random_y = np.random.randint(0, moire.size[1]-256)
                moire = moire.crop((random_x, random_y, random_x+256, random_y+256))
                clear = clear.crop((random_x, random_y, random_x+256, random_y+256))
        else:
            moire = moire.crop((6, 6, 1018, 1018))
            clear = clear.crop((6, 6, 1018, 1018))

        moire = self.transforms(moire)
        clear = self.transforms(clear)

        return moire, clear

    def __len__(self):
        return len(self.moire_images)

