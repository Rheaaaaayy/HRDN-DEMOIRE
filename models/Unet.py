import torch.nn as nn
import torch

# INPUT_SIZE = 572
# INPUT_CHANNEL = 1
# OUTPUT_SIZE = 388
# OUTPUT_CHANNEL = 2
INPUT_SIZE = 256
INPUT_CHANNEL = 3
OUTPUT_SIZE = 256
OUTPUT_CHANNEL = 3

class ConvBlock(nn.Module):
    def __init__(self,inplanes,planes):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,3,1,1)
        self.conv2 = nn.Conv2d(planes,planes,3,1,1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.relu(self.conv2(self.relu(self.conv1(x))))
        return out


class UpSample(nn.Module):
    def __init__(self,inplanes,planes):
        super(UpSample,self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2)
        self.convv = nn.Conv2d(inplanes,planes,1)

    def forward(self,x):
        out = self.convv(self.up_sample(x))
        return out


def copy_crop_concat(xl,xr):
        crop_idx = (xl.size(2)-xr.size(2))//2
        xl = xl[:,:,crop_idx:crop_idx+xr.size(2),crop_idx:crop_idx+xr.size(2)]
        return torch.cat((xl,xr),1)

class UNet(nn.Module):

    def __init__(self):
        super(UNet,self).__init__()
        self.left_conv1 = ConvBlock(INPUT_CHANNEL,64)
        self.left_conv2 = ConvBlock(64,128)
        self.left_conv3 = ConvBlock(128,256)
        self.left_conv4 = ConvBlock(256,512)
        self.left_conv5 = ConvBlock(512,1024)
        self.max_pool = nn.MaxPool2d(2,2)
        self.up1 = UpSample(1024,512)
        self.up2 = UpSample(512,256)
        self.up3 = UpSample(256,128)
        self.up4 = UpSample(128,64)
        self.right_conv1 = ConvBlock(1024,512)
        self.right_conv2 = ConvBlock(512,256)
        self.right_conv3 = ConvBlock(256,128)
        self.right_conv4 = ConvBlock(128,64)

    # def copy_crop_concat(self,xl,xr):
    #     crop_idx = (xl.size(2)-xr.size(2))//2
    #     xl = xl[:,:,crop_idx:crop_idx+xr.size(2),crop_idx:crop_idx+xr.size(2)]
    #     return torch.cat((xl,xr),1)


    def forward(self,x):
        x1 = self.left_conv1(x)
        x2 = self.left_conv2(self.max_pool(x1))
        x3 = self.left_conv3(self.max_pool(x2))
        x4 = self.left_conv4(self.max_pool(x3))
        x5 = self.left_conv5(self.max_pool(x4))

        x4_r = self.right_conv1(copy_crop_concat(x4,self.up1(x5)))
        x3_r = self.right_conv2(copy_crop_concat(x3,self.up2(x4_r)))
        x2_r = self.right_conv3(copy_crop_concat(x2,self.up3(x3_r)))
        x1_r = self.right_conv4(copy_crop_concat(x1,self.up4(x2_r)))

        output = nn.Conv2d(64,OUTPUT_CHANNEL,1)(x1_r)
        
        return output

if __name__ =="__main__":
    matrix = torch.randn(1,INPUT_CHANNEL,INPUT_SIZE,INPUT_SIZE)
    net = UNet()
    print(net)
    output = net(matrix)
    print(output.size())







