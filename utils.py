from torchvision import transforms as T
import numpy as np
import config
from torch.utils.data import SubsetRandomSampler

def get_transforms(train):
    trans_list = []
    trans_list.append(T.ToPILImage())
    trans_list.append(T.Resize((227,227)))
    trans_list.append(T.ToTensor())
    trans_list.append(T.Normalize(mean = [0.48423514409726315, 0.9742941201663617, 1.4248037412187176], std= [0.21358982510993574, 0.522832736875392, 0.992885390787591]))

    return T.Compose(trans_list)

 
def get_train_test_split(dataset_size):
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * config.TRAIN_SPLIT))
    if config.SHFFLE_DATASET == True:
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices) 
    train_indices, test_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler

def main():
    pass

if __name__ == '__main__':
    main()