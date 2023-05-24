import numpy as np
from sklearn.mixture import GaussianMixture
import sklearn.preprocessing as sk

np.random.seed(seed=13)

def normalized(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0: 
       return v
    return v / norm

class State:
    def __init__(self, isProductionState):
        # self.isEndState = isEndState
        self.isProductionState = isProductionState

        self.transition_matrix = None
        self.initial_distribution = None
        self.emission_distribution = None
        self.substates = []  # substates[0] = end state
    
    def add_substate(self, new_substate):
        self.substates.append(new_substate)
    
    def randomize(self):
        if self.isProductionState:
            # self.emission_distribution = GaussianMixture(n_components=1)
            self.emission_distribution = [.5,.5,1]
            self.emission_distribution = normalized(self.emission_distribution)

        else:
            dim = len(self.substates)
            self.transition_matrix = np.random.rand(dim, dim) # random transition matrix
            self.transition_matrix = sk.normalize(self.transition_matrix, norm='l1')
            self.initial_distribution = np.random.rand(dim) # random initial distribution
            self.initial_distribution = normalized(self.initial_distribution)
    
"""
    def load_transition_matrix(file_path):
    def load_initial_distribution(file_path):
    def load_emission_distribution(file_path):

    def load(file_path):
"""
    
