import numpy as np


class BernoulliNode:
    """Bernoulli node"""

    def __init__(self, p):
        """
        p: Bernoulli parameter
        """
        self.p = p
        self.mean = p
        self.var = p * (1 - p)
        self.val = (np.random.random() < p) * 1

    def sample(self):
        return (np.random.random() < self.p) * 1

    def resample(self):
        self.val = (np.random.random() < self.p) * 1

    def update(self, q):
        self.p = q
        self.mean = q
        self.var = q * (1 - q)
        self.val = (np.random.random() < q) * 1


class Graph:
    def __init__(self, nb_nodes, nodes=None, edges=None):
        self.nb_nodes = nb_nodes
        if nodes is None:
            self.nodes = [BernoulliNode(.5) for _ in range(nb_nodes)]
        else:
            self.nodes = nodes

        if edges is None:
            self.edges = np.zeros((nb_nodes, nb_nodes))
        else:
            self.edges = edges

    def update_node(self, i, *args):
        self.nodes[i].update(*args)

    def add_edge(self, i, j):
        self.edges[i, j] = 1

    def delete_edge(self, i, j):
        self.edges[i, j] = 0

    def grid_structure(self):
        """assumes number of nodes is a square number"""
        rn = int(np.sqrt(self.nb_nodes))
        for i in range(rn):
            for j in range(rn - 1):
                self.edges[i * rn + j, i * rn + j + 1] = 1
                self.edges[i + j * rn, i + (j + 1) * rn] = 1
