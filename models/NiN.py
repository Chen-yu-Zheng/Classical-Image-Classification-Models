import torch
from torch import nn


def nin_block(in_channels, out_channels, kernel_size= 1, stride= 1, padding= 0):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        nn.ReLU(inplace= True)
    )
    return block

class NiN(nn.Module):
    def __init__(self, num_classes):
        super(NiN, self).__init__()
        self.num_classes = num_classes

        self.network = nn.Sequential(
            #(1*227*227)
            nin_block(1, 96, kernel_size= 11, stride= 4, padding= 0), #(96*55*55)
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=3, stride= 2), #(96*27*27)

            nin_block(96,256,5,1,2), #(256*27*27)
            nn.Dropout(),
            nn.MaxPool2d(kernel_size= 3, stride= 2), #(256*13*13)

            nin_block(256,384,kernel_size= 3, stride= 1, padding= 1), #(384*13*13)
            nn.Dropout(),
            nn.MaxPool2d(kernel_size= 3, stride= 2), #(384,6,6)

            nin_block(384,self.num_classes,3,1,1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        out = self.network(x)
        out = out.view(out.shape[0], -1)
        return out

def main():
    net = NiN(num_classes= 10)
    print(net)
    img = torch.rand((2,1,227,227))
    print(img)
    out = net(img)
    print(out)

if __name__ == '__main__':
    main()