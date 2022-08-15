import torch
import torch.nn as nn
import numpy as np
import model.resnet as resnet



# C:\Users\lenovo/.cache\torch\hub\checkpoints\resnet34-333f7ec4.pth
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.backbone = resnet.resnet34(in_channels=3, pretrained=True, progress=True)

    def forward(self, x):
        resnet_out = self.backbone(x)
        # return resnet_out[2],resnet_out[3],resnet_out[4], resnet_out[5]
        return resnet_out
if __name__ == '__main__':

    data = torch.zeros(size=(32,3,256,256))
    ie = ImageEncoder()
    result = ie(data)
    I1,I2,I3,I4 = result
    print(I1.shape)
    print(I2.shape)
    print(I3.shape)
    print(I4.shape)

