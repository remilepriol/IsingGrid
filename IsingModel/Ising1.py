## How to represent my ising graph. If it is a grid, I can go for a matrix
# else I can go fr a list of neighbors. -> good for the algorithm
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)


def logistic(z):
    return 1 / (1 + np.exp(-z))


class IsingGrid(object):
    """ An Ising network on a 2D grid.
    Each node is either 1 either -1
    Two neighboring nodes have energy 0 if they have the same value, and energy a otherwise

    Attributes:
        n : size of the grid
        grid : contains the value of each spin
        a : second order factor of the energy
        b : contains the first order factor of the energy
    """

    def __init__(self, n=2, a=2, b=1):
        """Initialize a grid of size n with all nodes at -1"""
        self.n = n
        self.grid = -np.ones((n, n))
        self.a = a
        self.b = b

    def randomflips(self, theta):
        """set portion theta of the spin to 1"""
        mask = np.random.rand(self.n, self.n) < theta
        self.grid = 2 * mask - 1

    def energy(self):
        """ return the energy of the network at any given time"""
        # e1 = self.b * np.sum(self.grid)
        e1 = self.a * (np.sum(self.grid[1:, :] * self.grid[:-1, :] < 0))
        e1 += self.a * (np.sum(self.grid[:, 1:] * self.grid[:, :-1] < 0))
        return e1

    def savegrid(self, name):
        """plot the grid"""
        plt.imshow(self.grid, interpolation="nearest", cmap=plt.get_cmap('Greys'))
        plt.savefig('./images/' + name + '.pdf')

    def updategibbs(self, mask):
        """performs the gibbs sampling update for nodes that are not masked"""
        probagrid = (np.pad(self.grid[1:, :], ((0, 1), (0, 0)), mode='constant')
                     + np.pad(self.grid[:-1, :], ((1, 0), (0, 0)), mode='constant')
                     + np.pad(self.grid[:, 1:], ((0, 0), (0, 1)), mode='constant')
                     + np.pad(self.grid[:, :-1], ((0, 0), (1, 0)), mode='constant'))
        probagrid = logistic(self.a * probagrid)
        # probagrid = logistic(self.b * self.grid + self.a * probagrid)
        # We can add a b*grid term as a persistence factor that tends to preserve bits as they are
        # This is the typical update in statistical physics but it is not the actual gibbs update
        self.grid = mask * self.grid + (1 - mask) * (2 * (np.random.rand(self.n, self.n) < probagrid) - 1)

    def samplegibbs(self):
        """sample the grid according to joint distribution defined by a and b"""
        enrg = self.energy()
        energylist = [2 * enrg, enrg]
        epsilon = 0.001
        countiter = 0
        while energylist[-1] > 0 and abs(energylist[-2] / energylist[-1] - 1) > epsilon and countiter < 100:
            countiter += 1
            # mask half of the nodes, so that we only update nodes that are not touching each other.
            mask = np.fromfunction(lambda x, y: (x + y) % 2, (self.n, self.n))
            self.updategibbs(mask)
            # the update the other half
            self.updategibbs(1 - mask)
            energylist.append(self.energy())
        return energylist[1:]

    def meanfields(self):
        means = np.ones((self.n, self.n))
        sum_means_list = [2 * np.sum(means), np.sum(means)]
        # means_list = [means]
        epsilon = 0.001
        countiter = 0
        while abs(sum_means_list[-2] / sum_means_list[-1] - 1) > epsilon and countiter < 100:
            countiter += 1
            # update the means similarly to gibbs sampling, by taking means instead of the grid
            meangrid = (np.pad(means[1:, :], ((0, 1), (0, 0)), mode='constant') +
                        np.pad(means[:-1, :], ((1, 0), (0, 0)), mode='constant') +
                        np.pad(means[:, 1:], ((0, 0), (0, 1)), mode='constant') +
                        np.pad(means[:, :-1], ((0, 0), (1, 0)), mode='constant'))
            means = logistic(self.a * meangrid)
            sum_means_list.append(np.sum(means))
            # means_list.append(means)
        return means, sum_means_list[1:]

    def loopybelief(self):
        pass
