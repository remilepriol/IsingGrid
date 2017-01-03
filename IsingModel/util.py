import numpy as np
import matplotlib.pyplot as plt
import os


def gridmean(grid):
    x, y = np.shape(grid)
    return np.sum(np.absolute(grid)) / x / y


def logistic(z):
    return 1 / (1 + np.exp(-z))


def log_sum_exp(table):
    """
    Compute the log of sum of exp along the last column of table intelligently .

    log(sum_i exp(x_i)) = x_max + log(sum_i exp(x_i-x_max))
    """
    largest = np.amax(table, axis=-1)
    return largest + np.log(np.sum(np.exp(table - np.expand_dims(largest, axis=-1)), axis=-1))


def save_video(dirname, video):
    if os.path.exists(dirname):
        return
    else:
        os.makedirs(dirname)
    for t in range(video.shape[0]):
        plt.imshow(video[t], interpolation="nearest", cmap=plt.get_cmap('Greys'), vmin=-1, vmax=1)
        plt.savefig(dirname + "/frame" + str(t).zfill(4)+".pdf")