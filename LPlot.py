import numpy as np
import matplotlib.pyplot as plt
import math

# plot LLL add on 2022-01-17

def plot_loss(x, y):
    """
    plot the loss of training data

    input:
    x: number of epoch (type: int)
    y: value of loss (type: matrix)

    output:
    figure
    """

    n_epoch = np.arange(x) + 1
    loss = np.squeeze(y)

    #assert (y.shape == (1, x))

    plt.plot(n_epoch, loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.axis([1, x, 0, math.ceil(max(y))])
    plt.show()

def plot_BLEU(x, y):
    """
    plot the BLEU

    input:
    x: SNR matrix
    y: value of BLEU (type: matrix)

    output:
    figure
    """
    snr = np.array(x)
    BLEU = np.squeeze(y)

    #assert (y.shape == (1, x))

    plt.plot(snr, BLEU)
    plt.ylabel('BLEU score')
    plt.xlabel('SNR')
    plt.axis([0, max(snr), 0, math.ceil(max(y))])
    plt.show()




if __name__ == '__main__':
    t = np.load("saved_models/record1.npy")
    record_loss = np.array(t).reshape(1,20)  # for test
    x = np.arange(20)
    plt.plot(x, t)


    #bleu_score = [0.53691186, 0.66690329, 0.76383742, 0.84123271, 0.85956867, 0.88719197, 0.89458788]
    #SNR = [0, 3, 6, 9, 12, 15, 18]
    #plot_BLEU(SNR, bleu_score)
    t = np.load("saved_models/record1.npy")
    record_loss = np.array(t).reshape(1,20)  # for test
    x = np.arange(20)
    plt.plot(x, 2*t)
    plt.show()