import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class scale_layer(nn.Module):
    def __init__(self):
        super(scale_layer, self).__init__()
        alpha = torch.randn((3, 3), requires_grad=True)
        self.scale_alpha = torch.nn.Parameter(alpha)
        self.register_parameter("scale_alpha", self.scale_alpha)

    def forward(self, x):
        return x


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error) / X.size(0)
        return loss


class L1_Sobel_Loss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(L1_Sobel_Loss, self).__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(3, 1, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3, 1, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    def forward(self, source, target):

        edge_X_x = self.conv_op_x(source)
        edge_X_y = self.conv_op_y(source)
        edge_Y_x = self.conv_op_x(target)
        edge_Y_y = self.conv_op_y(target)
        edge_X = torch.sqrt(edge_X_x * edge_X_x + edge_X_y * edge_X_y)
        edge_Y = torch.sqrt(edge_Y_x * edge_Y_x + edge_Y_y * edge_Y_y)

        diff = torch.add(edge_X, -edge_Y)
        error = torch.sqrt(diff * diff)
        loss = torch.sum(error)
        loss /= source.size(0)

        return loss


class Weighted_Loss(nn.Module):
    def __init__(self):
        super(Weighted_Loss, self).__init__()
        self.Charbonnier_loss = L1_Charbonnier_loss()
        self.Sobel_Loss = L1_Sobel_Loss(device=torch.device('cuda'))

    def forward(self, X, Y):
        c_loss = self.Charbonnier_loss(X, Y)
        s_loss = self.Sobel_Loss(X, Y)
        loss = c_loss * 0.5 + s_loss * 0.5
        return loss


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

'''
if __name__ == '__main__':
    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_moire = Image.open("../image_moire.png").convert('RGB')
    w, h = image_moire.size
    image_moire = image_moire.crop((1 + int(0.15 * w), 1 + int(0.15 * h), int(0.85 * w), int(0.85 * h)))
    image_moire = T(image_moire)
    image_moire = image_moire.unsqueeze(0).cuda()

    image_clear = Image.open("../image_clear.png").convert('RGB')
    w, h = image_clear.size
    image_clear = image_clear.crop((1 + int(0.15 * w), 1 + int(0.15 * h), int(0.85 * w), int(0.85 * h)))
    image_clear = T(image_clear)
    image_clear = image_clear.unsqueeze(0).cuda()

    model = SimpleNet()
    model.cuda()
    criterion = Weighted_Loss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.00001,
        weight_decay=0.0001
    )


    for i in range(10):
        output_moire = model(image_moire)
        output_clear = model(image_clear)
        output_moire.retain_grad()

        output_moire.register_hook(lambda x: print(x))

        loss = criterion(output_moire, output_clear)
        print(loss)
        print("==========start backprops==========")
        loss.backward()
        print("==========end backprops==========")

        for name, i in model.named_parameters():
            print(name)
            print(i)
            print("grad is")
            print(i.grad)


        optimizer.step()
        optimizer.zero_grad()
'''



'''
    a = np.array([[[[1, 10, 2, 20],
                  [1, 1, 2, 2],
                  [3, 30, 4, 40],
                  [3, 3, 4, 4]],

                [[21, 10, 22, 20],
                 [21, 21, 22, 22],
                 [23, 30, 24, 40],
                 [23, 23, 24, 24]],

                [[31, 10, 32, 20],
                 [31, 31, 32, 32],
                 [33, 30, 34, 40],
                 [33, 33, 34, 34]]],
                  [[[1, 10, 2, 20],
                    [1, 1, 2, 2],
                    [3, 30, 4, 40],
                    [3, 3, 4, 4]],

                   [[21, 10, 22, 20],
                    [21, 21, 22, 22],
                    [23, 30, 24, 40],
                    [23, 23, 24, 24]],

                   [[31, 10, 32, 20],
                    [31, 31, 32, 32],
                    [33, 30, 34, 40],
                    [33, 33, 34, 34]]]],
                  dtype='float32')
'''

