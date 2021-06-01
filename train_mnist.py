import torch
from torch import nn
from torch.nn.utils import clip_grad
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import matplotlib.pyplot as plt
import numpy as np

import config
from models.AlexNet import AlexNet

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_train = FashionMNIST(config.ROOT_MNIST, train= True, transform= transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Resize((227,227)), transforms.ToTensor()]), download= False)
    '''
    print(dataset_train[0][0][0])
    print(len(dataset_train), len(dataset_test))
    plt.figure(1)
    plt.imshow(dataset_train[0][0][0].numpy())
    plt.show()
    '''
    train_iter = DataLoader(dataset_train, config.BATCH_SIZE, shuffle= True)
    
    net = AlexNet(in_channels= 1, num_classes= 10)
    #net.init_params()
    net.to(device)
    print(net)

    optimizer = optim.SGD(net.parameters(), config.LEARNING_RATE, weight_decay= config.WEIGHT_DECAY)
    loss_func = nn.CrossEntropyLoss()
    epoch_loss_history = []
    step_loss_history = []
    epoch_loss = 0

    net.train()
    '''
    plt.figure(2)
    plt.ion()
    plt.xlabel('step')
    plt.ylabel('loss')
    '''

    for epoch in range(config.EPOCH_NUM_MNIST):
        epoch_loss = 0.0
        for step, (imgs, labels) in enumerate(train_iter):
            #print(imgs.shape)
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = net(imgs)
            #print(out.shape)
            #print(labels.shape)

            loss = loss_func(out, labels)
            step_loss = loss.item()
            epoch_loss += step_loss
            step_loss_history.append(step_loss / config.BATCH_SIZE)

            if (step + 1) % 5 == 0:
                print('epoch: %d,step: %d, loss: %f' %(epoch+1, step+1, step_loss / config.BATCH_SIZE))
                '''
                plt.cla()
                plt.plot(np.arange(len(step_loss_history)) + 1, np.array(step_loss_history))
                plt.pause(0.01)
                '''
            optimizer.zero_grad()
            loss.backward()
            #clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()), max_norm= 35, norm_type= 2)
            optimizer.step()

        epoch_loss /= len(dataset_train)
        epoch_loss_history.append(epoch_loss)
        print('epoch %d, loss: %f' % (epoch + 1, epoch_loss))
    '''
    plt.ioff()
    plt.show()
    '''
    
    torch.save(net, config.RESULT + 'alexnet_no_pretrain_mnist.pkl')
    np.save(config.RESULT + 'step_loss_history_no_pretrain_mnist.npy', step_loss_history)
    np.save(config.RESULT + 'epoch_loss_history_no_pretrain_mnist.npy', epoch_loss_history)

    '''
    plt.figure(3)
    plt.plot(np.arange(1, config.EPOCH_NUM_MNIST + 1), np.array(epoch_loss_history))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure(4)
    plt.plot(np.arange(len(step_loss_history)) + 1, np.array(step_loss_history))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()
    '''

if __name__ == '__main__':
    main()