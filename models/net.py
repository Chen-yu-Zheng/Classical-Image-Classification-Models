from torchvision.models import alexnet
import torch.nn.functional as F
import torch
import torch.nn as nn


class Alexnet(nn.Module):
    def __init__(self, pretrain= False, num_classes = 21):
        super(Alexnet, self).__init__()
        self.alexnet = alexnet(pretrain)
        self._num_classes = num_classes
        self.cls_conv = nn.Conv2d(256, self._num_classes, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = x.cuda()
        feat = self.alexnet.features(x)
        feat = self.cls_conv(feat)
        logits = self.avg_pool(feat)
        # logits = self.model(x)
        
        return logits
    

