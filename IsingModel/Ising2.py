import numpy as np


def gridmean(grid):
    x, y = np.shape(grid)
    return np.sum(np.absolute(grid)) / x / y


def logistic(z):
    return 1 / (1 + np.exp(-z))


def log_sum_exp(args):
    """
    Compute the log of sum of exp of args intelligently.

    log(sum_i exp(x_i)) = x_j + log(sum_i exp(x_i-x_j))
    """
    values = np.asarray(list(args))
    largest = np.max(values, axis=-1)
    if largest <= float('-inf'):
        return float('-inf')
    return largest + np.log(np.sum(np.exp(values - largest)))


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
        __canonical_parameters: height*width*3 table containing canonical parameters of the model. Look at method energy()
    """

    def __init__(self, height, width):
        """Initialize a grid of size n with all nodes at -1"""
        self.height = height
        self.width = width
        self.linear_factor = np.zeros([height, width])
        self.vertical_correlation = np.zeros([height - 1, width])
        self.horizontal_correlation = np.zeros([height, width - 1])
        self.mean_parameters = np.zeros([height, width])
        self.__canonical_parameters = np.zeros([height, width, 5])

    def random_init(self):
        self.linear_factor = 2 * np.random.rand(self.height, self.width) - 1
        self.vertical_correlation = np.random.rand(self.height - 1, self.width)
        self.horizontal_correlation = np.random.rand(self.height, self.width - 1)

    def constant_init(self, linfac, vercor, horcor):
        self.linear_factor += linfac
        self.vertical_correlation += vercor
        self.horizontal_correlation += horcor

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
            enrg = -np.sum(self.linear_factor * grid)
            enrg -= 1 / 2 * np.sum(self.vertical_correlation * grid[1:, :] * grid[:-1, :])
            enrg -= 1 / 2 * np.sum(self.horizontal_correlation * grid[:, 1:] * grid[:, :-1])
            return enrg

    def __set_canonical(self):
        self.__canonical_parameters[:, :, 0] = self.linear_factor
        self.__canonical_parameters[:-1, :, 1] = self.vertical_correlation  # with lower neighbor
        self.__canonical_parameters[1:, :, 2] = self.vertical_correlation  # with upper neighbor
        self.__canonical_parameters[:, :-1, 3] = self.horizontal_correlation  # with right neighbor
        self.__canonical_parameters[:, 1:, 4] = self.horizontal_correlation  # with left neighbor

    def __sum_neighbors(self, grid):
        """
        return a new grid with the weighted sum of all neighbors on each point
        """
        return (np.pad(grid[1:, :] * self.vertical_correlation, ((0, 1), (0, 0)), mode='constant')
                + np.pad(grid[:-1, :] * self.vertical_correlation, ((1, 0), (0, 0)), mode='constant')
                + np.pad(grid[:, 1:] * self.horizontal_correlation, ((0, 0), (0, 1)), mode='constant')
                + np.pad(grid[:, :-1] * self.horizontal_correlation, ((0, 0), (1, 0)), mode='constant'))

    def __gibbs_update(self, grid, mask):
        """
        return a new grid where nodes that are not masked have been re sampled conditionally to their neighbors
        """
        # how to specify type in python 3 annotation
        probagrid = logistic(self.linear_factor + self.__sum_neighbors(grid))
        return mask * grid + (1 - mask) * (2 * (np.random.rand(self.height, self.width) < probagrid) - 1)

    def gibbs_sampling(self, grid):
        """starting from grid, performs gibbs sampling and return a new grid"""
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

    def meanfields(self, grid):
        """ update the mean_parameters with the mean field algorithm  """
        # the problem is non convex : different initialisation can return different outputs.
        # but we observe that different random initializations yields the same results.
        mean1 = gridmean(grid)
        sum_means_list = [mean1 + 1, mean1]
        epsilon = 1e-5
        countiter = 0
        while abs(sum_means_list[-2] - sum_means_list[-1]) > epsilon and countiter < 100:
            countiter += 1
            # update the means similarly to gibbs sampling
            grid = logistic(self.linear_factor + self.__sum_neighbors(grid))
            sum_means_list.append(gridmean(grid))
        self.mean_parameters = grid
        return sum_means_list[1:]

    def loopybelief(self, max_iter=25):
        # TODO
        # what is it supposed to return in the first place?
        messages = np.zeros([4, self.height, self.width, 2])
        # INGOING log-messages for each node
        # each message has two dimensional : it has a value for 1 and -1 at index 0 and 1 respectively.
        # each node emits 4 messages, one per neighbour:
        # 0 : bottom to top
        # 1 : top to bottom
        # 2 : left to right
        # 3 : right to left

        countiter = 0

        # stop condition is when we reach max_iter iterations.
        # Murphy et al. found that in average, max_iter = 15 is sufficient
        while countiter < max_iter:
            countiter += 1
            # first version, not correct, should update more frequently with weights
            upper = self.vertical_correlation * messages[:, :-1, :]
            lower = self.vertical_correlation * messages[:, 1:, :]
            righter = self.horizontal_correlation * messages[:, :, 1:]
            lefter = self.horizontal_correlation * messages[:, :, :-1]
            for k, newmessages, oldmessages in \
                    zip(range(4), [(upper, lower), (lower, upper), (righter, lefter), (lefter, righter)]):
                newmessages[k] = np.array(oldmessages[k + 1] + oldmessages[k + 2] + oldmessages[k + 3])
                a = 1
                newmessages[k, :, :, 0] = log_sum_exp(newmessages[k] - a * np.array([0, 1]))  # x=1
                newmessages[k, :, :, 1] = log_sum_exp(newmessages[k] - a * np.array([1, 0]))  # x=-1

        print(messages)

        self.mean_parameters = np.exp(np.sum(messages, axis=0))
        self.mean_parameters /= np.sum(self.mean_parameters, axis=-1)
