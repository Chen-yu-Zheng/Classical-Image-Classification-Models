import config
from torch.utils.data import DataLoader
from dataset import UCLandDataset
from utils import get_train_test_split

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
from torch import nn
from models.AlexNet import AlexNet
from models.VGG import VGG19

import sys

def main():
    dataset = UCLandDataset(config.ROOT, train= True)
    train_sampler = get_train_test_split(len(dataset))[0]
    train_iter = DataLoader(dataset, batch_size= config.BATCH_SIZE, sampler= train_sampler)

    '''
    for img, label in train_iter:
        print(img.size())
        print(label.size())
        break
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = AlexNet(in_channels= 3, num_classes= 21)
    #net = VGG19(num_classes= 21)
    #net.init_params()
    net = net.to(device)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr= config.LEARNING_RATE, weight_decay= config.WEIGHT_DECAY)
    loss_func = nn.CrossEntropyLoss()

    net.train()
    epoch_loss_history = []
    step_loss_history = []
    for epoch in range(config.EPOCH_NUM):
        epoch_loss = 0
        for step, (batch_imgs, batch_labels) in enumerate(train_iter):

            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            out = net(batch_imgs)
            '''
            print(out.shape)
            print(out.argmax(axis = 1).squeeze())
            print(batch_labels)
            print(batch_labels.shape)
            sys.exit()
            '''

            loss = loss_func(out, batch_labels)
            step_loss = loss.item()
            epoch_loss += step_loss
            step_loss_history.append(step_loss / config.BATCH_SIZE)

            if ((step + 1) % config.STEP == 0):
                print('epoch: %d,step %d, loss: %f' % (epoch + 1,step + 1, step_loss / config.BATCH_SIZE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss /= (len(dataset) * config.TRAIN_SPLIT)
        epoch_loss_history.append(epoch_loss)
        print('epoch %d, loss: %f' % (epoch + 1, epoch_loss))
        
    step_loss_history = np.array(step_loss_history)
    epoch_loss_history = np.array(epoch_loss_history)
    np.save(config.RESULT + 'step_loss_history_no_pretrain_ucm.npy', step_loss_history)
    np.save(config.RESULT + 'epoch_loss_history_no_pretrain_ucm.npy', epoch_loss_history)

    torch.save(net, config.RESULT + 'alexnet_no_pretrain_ucm.pkl')
    #torch.save(net, config.RESULT + 'vgg_no_pretrain_ucm.pkl')
           
if __name__ == '__main__':
    main()

