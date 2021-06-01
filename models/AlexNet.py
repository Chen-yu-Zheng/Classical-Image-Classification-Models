import torch
from torch import nn
from torch.nn import init

class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        #img_size:(3,227,227)
        super(AlexNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= self.in_channels, out_channels= 96, kernel_size= 11, stride= 4, padding= 0),
            nn.ReLU(inplace= True),
            #nn.LocalResponseNorm(size= 5, alpha= 0.0001, beta= 0.75, k= 2),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size= 5, stride= 1, padding= 2),
            nn.ReLU(inplace= True),
            #nn.LocalResponseNorm(size= 5, alpha= 0.0001, beta= 0.75, k= 2),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels= 256, out_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 384, out_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 384, out_channels= 256, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.Dense = nn.Sequential(
            nn.Dropout(p= 0.5),
            nn.Linear(in_features= 256*6*6, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p= 0.5),
            nn.Linear(in_features= 4096, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Linear(4096,self.num_classes)
        )

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean= 0.0, std = 0.01)
            else:
                if ('layer2' in name) or ('layer3.2' in name) or ('layer3.4' in name) or ('Dense' in name):
                    init.constant_(param, 1)
                else:
                    init.constant_(param, 0)
                

    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.shape[0], -1)
        out = self.Dense(out)
        return out

def main():
    net = AlexNet(in_channels=3, num_classes= 21)
    net.init_params()

    img = torch.rand((1,3,227,227))
    out = net(img)
    print(out)

    for name, param in net.named_parameters():
        print(name)

if __name__ == '__main__':
    main()