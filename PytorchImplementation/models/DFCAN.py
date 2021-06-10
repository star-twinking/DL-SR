import torch
import torch.nn as nn
# from .common import fftshift2d
import torch.fft
from torchsummary import summary

def fftshift2d(img, size_psc=128):
    bs,ch, h, w = img.shape
    fs11 = img[:,:, h//2:, w//2:]
    fs12 = img[:,:, h//2:, :w//2]
    fs21 = img[:,:, :h//2, w//2:]
    fs22 = img[:,:, :h//2, :w//2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    # output = tf.image.resize_images(output, (size_psc, size_psc), 0)
    return output

class FCAB(nn.Module):
    '''
    Fourier channel attention block
    '''
    def __init__(self):
        super(FCAB, self).__init__()

        self.conv_gelu1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 缩小两个
            nn.GELU()
        )
        self.conv_gelu2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_relu1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(4,64,kernel_size=1,stride=1, padding=0),
            nn.Sigmoid()
        )
        # self.conv = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=1)
        # self.sig = nn.Sigmoid()

    def forward(self, x, gamma=0.8):
        x0 = x
        x = self.conv_gelu1(x)
        x = self.conv_gelu2(x)
        x1 = x
        x = torch.fft.fftn(x, dim=(2,3)) # 这里的dimension要注意
        x = torch.pow(torch.abs(x) + 1e-8, gamma)  # abs
        x =fftshift2d(x)
        x = self.conv_relu1(x)
        x = self.avgpool(x)
        x = self.conv_relu2(x)
        x = self.conv_sig(x)
        # print(x.shape)
        # print(x1.shape)
        x = x1*x
        output = x+x0
        return output


class ResidualGroup(nn.Module):
    def __init__(self, n_fcab=4):
        super(ResidualGroup,self).__init__()
        FCABs = []
        for _ in range(n_fcab):
            FCABs.append(FCAB())
        self.FCABs = nn.Sequential(*FCABs) #变成顺序，*会将原来的模型变成一个一个层

    def forward(self, x):
        x0=x
        x = self.FCABs(x)
        output = x+x0
        return output

class DFCAN(nn.Module):
    def __init__(self, input_shape, scale=2, size_psc=128):
        super(DFCAN, self).__init__()
        self.conv_gelu1 = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=3,stride=1, padding=1),
            nn.GELU()
        )

        n_residualgroups=4
        ResidualGroups =[]
        for _ in range(n_residualgroups):
            ResidualGroups.append(ResidualGroup(n_fcab=4))
        self.res = nn.Sequential(*ResidualGroups)

        self.conv_gelu2 = nn.Sequential(
            nn.Conv2d(64, 64*(scale**2), kernel_size=3,stride=1,padding=1),
            nn.GELU()
        )
        self.pixelshuffle =nn.PixelShuffle(scale)
        self.conv_sig = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_gelu1(x)
        x = self.res(x)
        x = self.conv_gelu2(x)
        x = self.pixelshuffle(x) #upsampling
        output = self.conv_sig(x)
        return output

# 测试模型
if __name__ == '__main__':
    x = torch.rand(1,6,128,128)
    model = DFCAN(input_shape=x.size()[1])
    y = model(x)
    print('Output shape:', y.shape)
    # 转为GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    if torch.cuda.is_available():
        model.cuda()
    summary(model,input_size=(6,128,128))