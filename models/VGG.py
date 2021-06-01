import torch
from torch import nn
from torch.nn import init

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        #image size (3,224,224)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 3, padding= 1, stride= 1), #(64,224,224)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3, padding= 1, stride= 1),#(64,224,224)
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2), #(64,112,112)

            nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3, padding= 1, stride= 1), #(128,112,112)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3, padding= 1, stride= 1), #(128,112,112)
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2), #(128,56,56)

            nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 3, padding= 1, stride= 1), #(256,56,56)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size= 3, padding= 1, stride= 1), #(256,56,56)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size= 3, padding= 1, stride= 1), #(256,56,56)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size= 3, padding= 1, stride= 1), #(256,56,56)
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2), #(256,28,28)

            nn.Conv2d(in_channels= 256, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2), #(512,14,14)

            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, padding= 1, stride= 1), #(512,28,28)
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2) #(512,7,7)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features= 512*7*7, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p= 0.5),

            nn.Linear(in_features= 4096, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p= 0.5),

            nn.Linear(in_features= 4096, out_features= num_classes)
        )
    
    def forward(self, img):
        out = self.conv_pool(img)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    vgg19 = VGG19(21)
    print(vgg19)
    imgs = torch.ones((2,3,224,224))
    out = vgg19(imgs)
    print(out)