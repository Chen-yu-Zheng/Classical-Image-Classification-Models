import config
import numpy as np 
import matplotlib.pyplot as plt

def main():
    step_loss_history = np.load(config.RESULT + 'step_loss_history_no_pretrain_ucm.npy')

    epoch_loss_history = np.load(config.RESULT + 'epoch_loss_history_no_pretrain_ucm.npy')
    plt.figure(1)
    plt.plot(np.arange(len(epoch_loss_history)) + 1, np.array(epoch_loss_history))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    

    plt.figure(2)
    plt.plot(np.arange(len(step_loss_history)) + 1, np.array(step_loss_history))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    main()