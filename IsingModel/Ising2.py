from IsingModel.util import *
import numpy as np


class IsingGrid(object):
    """
    An Ising network on a 2D grid.
    Each node is either 1 either -1.
    Two neighboring nodes i and j have energy -a_ij if they have the same value, and energy a_ij otherwise.
    Energy(x) = - 1/2*sum_i(b_i*x_i) - 1/2*sum_ij(x_i*a_ij*x_j )
    probability(x) is proportional to exp(-E(x)), hence x_i will tend to have the same sign as b_i
    and for a_ij>0, x_i and x_j will tend to have the same sign.

    Attributes:
        height, width : size of the grid
        observations : grid where observed points have value +1 or -1, latent variables have value 0
        linear_factors : canonical parameters b_i
        vertical correlations : canonical parameters a_ij
        linea correlations : canonical parameters a_ij
        mean_parameters : estimate of the mean value for each node
    """

    def __init__(self, height, width):
        """Initialize a grid of size n with all nodes at -1"""
        self.height = height
        self.width = width
        self.observations = np.zeros([height, width])
        self.linear_factors = np.zeros([height, width])
        self.vertical_correlations = np.zeros([height - 1, width])
        self.horizontal_correlations = np.zeros([height, width - 1])
        self.mean_parameters = np.zeros([height, width])
        self.__canonical_parameters = np.zeros([height, width, 5])

    def random_init(self):
        self.linear_factors = 2 * np.random.rand(self.height, self.width) - 1
        self.vertical_correlations = np.random.rand(self.height - 1, self.width)
        self.horizontal_correlations = np.random.rand(self.height, self.width - 1)

    def constant_init(self, linfac, vercor, horcor):
        self.linear_factors += linfac
        self.vertical_correlations += vercor
        self.horizontal_correlations += horcor

    def observe(self, grid):
        if self.is_correct(grid):
            self.observations = grid

    def random_grid(self, theta):
        """return a well sized grid with portion theta of the pixels set to 1"""
        return 2 * (np.random.rand(self.height, self.width) < theta) - 1

    def is_correct(self, grid):
        return grid.shape == (self.height, self.width)

    def trim(self, grid):
        return grid[:self.height, :self.width]

    def energy(self, grid):
        """
        Return the energy of a grid according to the model
        E = -sum_i(b_i*x_i) - 1/2*sum_ij(x_i*a_ij*x_j )
        """
        if self.is_correct(grid):
            enrg = -np.sum(self.linear_factors * grid)
            enrg -= 1 / 2 * np.sum(self.vertical_correlations * grid[1:, :] * grid[:-1, :])
            enrg -= 1 / 2 * np.sum(self.horizontal_correlations * grid[:, 1:] * grid[:, :-1])
            return enrg

    def __set_canonical(self):
        self.__canonical_parameters[:, :, 0] = self.linear_factors
        self.__canonical_parameters[:-1, :, 1] = self.vertical_correlations  # with lower neighbor
        self.__canonical_parameters[1:, :, 2] = self.vertical_correlations  # with upper neighbor
        self.__canonical_parameters[:, :-1, 3] = self.horizontal_correlations  # with right neighbor
        self.__canonical_parameters[:, 1:, 4] = self.horizontal_correlations  # with left neighbor

    def __sum_neighbors(self, grid):
        """
        return a new grid with the weighted sum of all neighbors on each point
        """
        return (np.pad(grid[1:, :] * self.vertical_correlations, ((0, 1), (0, 0)), mode='constant')
                + np.pad(grid[:-1, :] * self.vertical_correlations, ((1, 0), (0, 0)), mode='constant')
                + np.pad(grid[:, 1:] * self.horizontal_correlations, ((0, 0), (0, 1)), mode='constant')
                + np.pad(grid[:, :-1] * self.horizontal_correlations, ((0, 0), (1, 0)), mode='constant'))

    def __gibbs_update(self, grid, mask):
        """
        return a new grid where nodes that are not masked have been re sampled conditionally to their neighbors
        """
        # how to specify type in python 3 annotation
        probagrid = logistic(self.linear_factors + self.__sum_neighbors(grid))
        return mask * grid + (1 - mask) * (2 * (np.random.rand(self.height, self.width) < probagrid) - 1)

    def gibbs_sampling(self, grid):
        """starting from grid, performs gibbs sampling and return a new grid"""
        if not self.is_correct(grid):
            return
        enrg = self.energy(grid)
        energylist = [2 * enrg, enrg]
        epsilon = 1e-5
        countiter = 0
        while abs(energylist[-2] - energylist[-1]) > epsilon and countiter < 100:
            countiter += 1
            # mask half of the nodes in a checkerboard pattern
            mask = np.fromfunction(lambda x, y: (x + y) % 2, (self.height, self.width))
            # update nodes that are not touching each other
            grid = self.__gibbs_update(grid, mask)
            # update the other half
            grid = self.__gibbs_update(grid, 1 - mask)
            energylist.append(self.energy(grid))
        return grid, energylist[1:]

    def gibbs_video(self, grid, time_max=100):
        """starting from grid, performs gibbs sampling and return the whole path taken by the grid"""
        if not self.is_correct(grid):
            return
        enrg = self.energy(grid)
        energylist = [2 * enrg, enrg]
        video = np.empty([time_max, self.height, self.width])
        for t in range(time_max):
            # mask half of the nodes in a checkerboard pattern
            mask = np.fromfunction(lambda x, y: (x + y) % 2, (self.height, self.width))
            # update nodes that are not touching each other
            grid = self.__gibbs_update(grid, mask)
            # update the other half
            grid = self.__gibbs_update(grid, 1 - mask)
            video[t, :, :] = grid
            energylist.append(self.energy(grid))
        return video, energylist[1:]

    def meanfields(self, initial_grid, max_iter=100):
        """
        Update the mean parameters with the mean field algorithm
        Since the problem is non convex, different initializations can return different outputs.
        :param initial_grid: starting point, best results with zeros
        :param max_iter:
        :return:
        """
        #
        grid = self.observations + (1 - np.absolute(self.observations)) * initial_grid
        enrg1 = self.energy(grid)
        epsilon = 1e-5
        means_energy = [enrg1 + 2*epsilon, enrg1]
        countiter = 0
        while abs(means_energy[-2] - means_energy[-1]) > epsilon and countiter < max_iter:
            countiter += 1
            # update the means similarly to gibbs sampling
            grid = 2 * logistic(self.linear_factors + self.__sum_neighbors(grid)) - 1
            grid = self.observations + (1 - np.absolute(self.observations)) * grid
            means_energy.append(self.energy(grid))
        self.mean_parameters = grid
        return means_energy[1:]

    @staticmethod
    def __repeat_symmetric(grid):
        tmp = np.expand_dims(grid, axis=-1)
        return np.concatenate((tmp, - tmp), axis=-1)

    def __messages2means(self,messages):
        self.mean_parameters = np.product(messages, axis=0)
        self.mean_parameters = self.mean_parameters[:, :, 0] / np.sum(self.mean_parameters, axis=-1)
        self.mean_parameters = 2 * self.mean_parameters - 1

    def loopybelief(self, max_iter=25,damping=0.5):
        """
        update the mean_parameters field with the probability given by the sum product algorithm
        :param max_iter:
        :return: means_energy: the sequence of mean parameters energy
        """
        messages = np.ones([5, self.height, self.width, 2])
        # INGOING messages for each node
        # each message has two dimensional : it has a value for 1 and -1 at index 0 and 1 respectively.
        # each node emits 4 messages, one per neighbour:
        # 0 : bottom to top
        # 1 : left to right
        # 2 : top to bottom
        # 3 : right to left
        # 4 : log-potentials in each point
        # potentials are stored in an additional channel
        # if +1 is observed, set potential for -1 to 0
        messages[4, :, :, 1] = (1 - (self.observations == 1)) * self.linear_factors / 2
        # if -1 is observed, set potential for +1 to 0
        messages[4, :, :, 0] = (1 - (self.observations == -1)) * (-self.linear_factors / 2)

        upper = messages[:, :-1, :]
        lower = messages[:, 1:, :]
        righter = messages[:, :, 1:]
        lefter = messages[:, :, :-1]

        correlations = [self.vertical_correlations, self.horizontal_correlations, self.vertical_correlations,
                        self.horizontal_correlations]
        means_energy = []
        # we stop when we reach max_iter iterations.
        # Murphy et al. found that in average, max_iter = 15 is sufficient
        for _ in range(max_iter):
            old_messages = messages.copy()
            old_upper = old_messages[:, :-1, :]
            old_lower = old_messages[:, 1:, :]
            old_righter = old_messages[:, :, 1:]
            old_lefter = old_messages[:, :, :-1]
            for k, (newmessages, oldmessages) in zip(range(4),
                        [(upper, old_lower), (righter, old_lefter), (lower, old_upper), (lefter, old_righter)]):
                # sum of messages coming to the source node, except the one coming from the destination node
                newmessages[k] = oldmessages[(k - 1)] * oldmessages[k] * oldmessages[(k + 1) % 4] * oldmessages[4]
                corr = self.__repeat_symmetric(correlations[k] / 2)
                # x_i=+1 in newmessages[k, :, :, 0]
                tmp = np.sum(newmessages[k] * np.exp(corr), axis=-1)
                # x_i=-1 in newmessages[k, :, :, 1]
                newmessages[k, :, :, 1] = np.sum(newmessages[k] * np.exp(- corr), axis=-1)
                newmessages[k, :, :, 0] = tmp
                # normalization of messages to 1
                newmessages[k] /= np.expand_dims(np.sum(newmessages[k], axis=-1), axis=-1)
            # damping
            messages[:4] = damping*messages[:4] + (1-damping)*old_messages[:4]
            upper = messages[:, :-1, :]
            lower = messages[:, 1:, :]
            righter = messages[:, :, 1:]
            lefter = messages[:, :, :-1]
            self.__messages2means(messages)
            means_energy.append(self.energy(self.mean_parameters))
        self.__messages2means(messages)
        return means_energy

