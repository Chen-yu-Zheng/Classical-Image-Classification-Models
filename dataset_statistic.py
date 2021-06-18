import config
from dataset import UCLandDataset

def main():
    dataset = UCLandDataset(config.ROOT, train= True)
    means = []
    stds = []
    sum = 0
    std = 0
    for channel in range(3):
        for i in range(len(dataset)):
            img = dataset[i][0][channel]
            sum += img.sum().item()
        mean = sum / (len(dataset)*227*227)
        means.append(mean)

        for i in range(len(dataset)):
            img = dataset[i][0][channel]
            std += ((img - mean) * (img - mean)).sum().item()
        std = std / (len(dataset)*227*227)
        std = std ** 0.5
        stds.append(std)
    
    print(means)
    print(stds)

if __name__ == '__main__':
    main()
    #mean = [0.48423514409726315, 0.9742941201663617, 1.4248037412187176]
    #std = [0.21358982510993574, 0.522832736875392, 0.992885390787591]