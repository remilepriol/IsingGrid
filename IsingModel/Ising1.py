import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)


def logistic(z):
    return 1 / (1 + np.exp(-z))


def log_sum_exp(args):
    '''
    Compute the log of sum of exp of args intelligently.

    log(sum_i exp(x_i)) = x_j + log(sum_i exp(x_i-x_j))
    '''
    values = np.asarray(list(args))
    largest = np.max(values, axis=-1)
    if largest <= float('-inf'):
        return float('-inf')
    return largest + np.log(np.sum(np.exp(values - largest)))


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

    def __init__(self, n=2, a=2, b=0):
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

    def mean(self):
        return np.sum(self.grid)/self.n**2

    def savegrid(self, name):
        """plot the grid"""
        plt.title(name)
        plt.imshow(self.grid, interpolation="nearest", cmap=plt.get_cmap('Greys'),vmin=-1,vmax=1)
        #plt.savefig('./images/' + name + '.pdf')

    def crossconvol(self):
        """return the convolution of the grid with a cross,
        ie return a grid with the sum of all neighbors on each point
        00000
        00100
        01010
        00100
        00000
        """
        return (np.pad(self.grid[1:, :], ((0, 1), (0, 0)), mode='constant')
                + np.pad(self.grid[:-1, :], ((1, 0), (0, 0)), mode='constant')
                + np.pad(self.grid[:, 1:], ((0, 0), (0, 1)), mode='constant')
                + np.pad(self.grid[:, :-1], ((0, 0), (1, 0)), mode='constant'))

    def updategibbs(self, mask):
        """performs the gibbs sampling update for nodes that are not masked"""
        probagrid = logistic(self.b + self.a * self.crossconvol())
        # probagrid = logistic(self.b * self.grid + self.a * probagrid)
        # We can add a b*grid term as a persistence factor that tends to preserve bits as they are
        # This is the typical update in statistical physics but it is not the actual gibbs update
        self.grid = mask * self.grid + (1 - mask) * (2 * (np.random.rand(self.n, self.n) < probagrid) - 1)

    def samplegibbs(self):
        """sample the grid according to the joint distribution defined by a and b"""
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
        """update the grid that stores the means in place """
        mean1 = self.mean()
        sum_means_list = [mean1 + 1, mean1]
        epsilon = 0.00001
        countiter = 0
        while abs(sum_means_list[-2] - sum_means_list[-1]) > epsilon and countiter < 100:
            countiter += 1
            # update the means similarly to gibbs sampling, by taking means instead of the grid
            self.grid = logistic(self.b + self.a * self.crossconvol())
            sum_means_list.append(self.mean())
        return sum_means_list[1:]

    def loopybelief(self, max_iter=25):
        n = self.n
        messages = np.zeros(4, n, n, 2)
        # INGOING log-messages for each node
        # each message has two dimensional : it has a value for 1 and -1 at index 0 and 1 respectively.
        # each node emits 4 messages, one per neighbour:
        # 0 : bottom to top
        # 1 : top to bottom
        # 2 : left to right
        # 3 : right to left

        upper = messages[:, :-1, :]
        lower = messages[:, 1:, :]
        righter = messages[:, :, 1:]
        lefter = messages[:, :, :-1]

        countiter = 0

        # stop condition is when we reach max_iter iterations.
        # Murphy et al. found that in average, max_iter = 15 is sufficient
        while countiter < max_iter:
            countiter += 1
            for k, newmessages, oldmessages in \
                    zip(range(4), [(upper, lower), (lower, upper), (righter, lefter), (lefter, righter)]):
                newmessages[k] = np.array(oldmessages[k + 1] + oldmessages[k + 2] + oldmessages[k + 3])
                newmessages[k, :, :, 0] = log_sum_exp(newmessages[k] - self.a * np.array([0, 1]))  # x=1
                newmessages[k, :, :, 1] = log_sum_exp(newmessages[k] - self.a * np.array([1, 0]))  # x=-1

        print(messages)

        self.grid = np.product(messages, axis=0)
        self.grid /= np.sum(self.grid, axis=-1)

