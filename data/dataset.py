import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms

class MoireData(data.Dataset):

    def __init__(self, root, transform=None):
        moire_data_root = os.path.join(root, "moire")
        clear_data_root = os.path.join(root, "clear")

        image_names = os.listdir(clear_data_root)
        self.moire_images = [os.path.join(moire_data_root, x) for x in image_names]
        self.clear_images = [os.path.join(clear_data_root, x) for x in image_names]

        if transform is not None:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]

        moire = Image.open(moire_img_path).convert('RGB')
        clear = Image.open(clear_img_path).convert('RGB')

        random_x =np.random.randint(0, moire.size[0]-256)
        random_y = np.random.randint(0, moire.size[1]-256)
        moire = moire.crop((random_x, random_y, random_x+256, random_y+256))
        clear = clear.crop((random_x, random_y, random_x+256, random_y+256))
        print(moire.size)

        moire = self.transforms(moire)
        clear = self.transforms(clear)


        return moire, clear

    def __len__(self):
        return len(self.moire_images)

