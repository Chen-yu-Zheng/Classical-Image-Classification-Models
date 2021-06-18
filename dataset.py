import config
import torch
from torch.utils.data import Dataset
import os
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from utils import get_transforms

class UCLandDataset(Dataset):
    def __init__(self, root, train = True):
        self.root = root
        self.transforms = get_transforms(train)
        self.imgs = []
        self.labels = []

        root_dir = os.listdir(root)
        label = 0
        for son_dir in root_dir:
            son_dir_imgs = os.listdir(os.path.join(self.root, son_dir))
            for i in range(len(son_dir_imgs)):
                self.imgs.append(os.path.join(self.root, son_dir, son_dir_imgs[i]).replace('\\', '/'))
            self.labels += len(son_dir_imgs) * [label]
            label += 1
        #print(self.imgs)
        #print(self.labels)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = imread(img_path)
        #img = imread(img_path).astype(np.float32, copy= False)
        label = self.labels[index]

        #print(img)
        #img -= self.mean
        #img /= self.std
        '''
        plt.figure(1)
        plt.imshow(img)
        plt.show()
        '''
        
        #img = np.transpose(img, axes= [2,0,1])
        #img = torch.tensor(img, dtype= torch.float32)
        #label = torch.tensor(label, dtype= torch.float32)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


def main():
    dataset = UCLandDataset(config.ROOT, train= True)
    print(len(dataset))
    '''
    img, label = dataset[0]
    #img = img.numpy()
    #img = np.transpose(img, axes = [1,2,0])
    print(img)
    print(label)
    plt.figure(2)
    plt.imshow(img)
    plt.show()
    '''

if __name__ == '__main__':
    main()