import os
import os.path
from PIL import Image
from torch.utils import data
from torchvision import transforms


# Crop input as Sun et al.
def default_loader(path):
  img = Image.open(path).convert('RGB')
  w, h = img.size
  region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
  return region


class Val_MoireData(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
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
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        label = self.labels[index]
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]

        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)

        moire = self.transforms(moire)
        clear = self.transforms(clear)

        return moire, clear, label


    def __len__(self):
        return len(self.moire_images)
