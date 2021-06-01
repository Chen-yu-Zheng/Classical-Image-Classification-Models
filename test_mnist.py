import config
from models.AlexNet import AlexNet
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms

import numpy as np

import torch
from torch import nn

def main():
    dataset_train = FashionMNIST(config.ROOT_MNIST, train= True, transform= transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Resize((227,227)), transforms.ToTensor()]), download= False)
    dataset_test = FashionMNIST(config.ROOT_MNIST, train= False, transform= transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Resize((227,227)), transforms.ToTensor()]), download= False)
    train_iter = DataLoader(dataset_train, config.BATCH_SIZE, shuffle= True)
    test_iter = DataLoader(dataset_test, config.BATCH_SIZE, shuffle= False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torch.load(config.RESULT + 'alexnet_no_pretrain_mnist.pkl')
    net = net.to(device)
    print(net)

    net.eval()
    acc_train = 0.0
    for (imgs, labels) in train_iter:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = net(imgs)
        out = out.argmax(axis = 1)
        acc_train += (out == labels).sum().item()
    acc_train /= len(dataset_train)
    print('Acc on train dataset: %f' % acc_train)

    acc_test = 0.0
    for (imgs, labels) in test_iter:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = net(imgs)
        out = out.argmax(axis = 1)
        acc_test += (out == labels).sum().item()
    acc_test /= len(dataset_test)
    print('Acc on test dataset: %f' % acc_test)

    '''wihout dropout and weight-decay
    Acc on train dataset: 0.972017
    Acc on test dataset: 0.921200
    '''

    '''without dropout
    Acc on train dataset: 0.958250
    Acc on test dataset: 0.918700
    '''


if __name__ == '__main__':
    main()