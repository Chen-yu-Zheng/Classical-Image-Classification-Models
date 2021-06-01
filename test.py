import config
from models.AlexNet import AlexNet
from torch.utils.data import DataLoader
from dataset import UCLandDataset
from utils import get_train_test_split

import numpy as np

import torch
from torch import nn

def main():
    dataset = UCLandDataset(config.ROOT, train= False)
    train_sampler = get_train_test_split(len(dataset))[0]
    train_iter = DataLoader(dataset, batch_size= config.BATCH_SIZE, sampler= train_sampler)

    test_sampler = get_train_test_split(len(dataset))[1]
    test_iter = DataLoader(dataset, batch_size= config.BATCH_SIZE, sampler= test_sampler)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torch.load(config.RESULT + 'alexnet_no_pretrain_ucm.pkl')
    net = net.to(device)
    print(net)

    net.eval()
    acc_train = 0.0
    for batch_imgs, batch_labels in train_iter:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = net(batch_imgs)
        pred = out.argmax(axis = 1)
        acc_train += (batch_labels == pred).sum().item()
    acc_train /= float(len(dataset) * config.TRAIN_SPLIT)
    print('Acc on train dataset: %f' % (acc_train))

    acc_test = 0.0
    for batch_imgs, batch_labels in test_iter:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = net(batch_imgs)
        pred = out.argmax(axis = 1)
        acc_test += (batch_labels == pred).sum().item()
    acc_test /= float(len(dataset) * (1 - config.TRAIN_SPLIT))
    print('Acc on test dataset: %f' % (acc_test))

if __name__ == '__main__':
    main()