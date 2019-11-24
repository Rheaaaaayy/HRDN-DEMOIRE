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
    def __init__(self):
        super(L1_Sobel_Loss, self).__init__()

        self.eps = 1e-6
        self.conv_op_x = nn.Conv2d(1, 1, 3, bias=False)
        self.conv_op_y = nn.Conv2d(1, 1, 3, bias=False)
        sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, source, target):

        edge_detect_X = torch.ones((source.size(0), 1, source.size(2)-2, source.size(3)-2))
        edge_detect_Y = torch.ones((source.size(0), 1, source.size(2) - 2, source.size(3) - 2))

        for c in range(3):
            X = source[:, c:c+1, :, :]
            Y = target[:, c:c+1, :, :]
            # print("X_{}_size:{}".format(c, X.size()))
            edge_X_x = self.conv_op_x(X)
            edge_X_y = self.conv_op_y(X)

            edge_Y_x = self.conv_op_x(Y)
            edge_Y_y = self.conv_op_y(Y)

            edge_X = torch.sqrt(edge_X_x * edge_X_x + edge_X_y * edge_X_y)
            edge_Y = torch.sqrt(edge_Y_x * edge_Y_x + edge_Y_y * edge_Y_y)

            edge_detect_X = torch.cat((edge_detect_X, edge_X), dim=1)
            edge_detect_Y = torch.cat((edge_detect_Y, edge_Y), dim=1)

        edge_detect_X = edge_detect_X[:, 1:, :, :]
        edge_detect_Y = edge_detect_Y[:, 1:, :, :]

        diff = torch.add(edge_detect_X, -edge_detect_Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error) / source.size(0)

        # plt.subplot(121)
        # plt.imshow(edge_detect_X.squeeze().permute(1, 2, 0))
        # plt.subplot(122)
        # plt.imshow(edge_detect_Y.squeeze().permute(1, 2, 0))
        # plt.show()

        return loss


class Weighted_Loss(nn.Module):
    def __init__(self):
        super(Weighted_Loss, self).__init__()
        self.Charbonnier_loss = L1_Charbonnier_loss()
        self.Sobel_Loss = L1_Sobel_Loss()

    def forward(self, X, Y):
        c_loss = self.Charbonnier_loss(X, Y)
        s_loss = self.Sobel_Loss(X, Y)
        loss = c_loss * 0.5 + s_loss * 0.5
        return loss


if __name__ == '__main__':
    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_moire = Image.open("image_moire.png").convert('RGB')
    w, h = image_moire.size
    image_moire = image_moire.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    image_moire = T(image_moire)
    image_moire = image_moire.unsqueeze(0)

    image_clear = Image.open("image_clear.png").convert('RGB')
    w, h = image_clear.size
    image_clear = image_clear.crop((1 + int(0.15 * w), 1 + int(0.15 * h), int(0.85 * w), int(0.85 * h)))
    image_clear = T(image_clear)
    image_clear = image_clear.unsqueeze(0)


    criterion = Weighted_Loss()
    loss = criterion(image_moire, image_clear)
    print(loss)

    for name, i in criterion.named_parameters():
        print(name, i)



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


