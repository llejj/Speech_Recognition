import numpy as np
from sklearn.mixture import GaussianMixture
import sklearn.preprocessing as sk

np.random.seed(seed=10)

def normalized(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0: 
       return v
    return v / norm

class State:
    def __init__(self, isProductionState, parent=None, state_index=None):
        # self.isEndState = isEndState
        self.isProductionState = isProductionState
        self.parent = parent
        self.state_index = state_index

        self.transition_matrix = None
        self.initial_distribution = None
        self.emission_distribution = None

        self.substates = []  # substates[0] = end state
    
    def add_substate(self, new_substate):
        self.substates.append(new_substate)        

    def randomize(self):
        if self.isProductionState:
            # self.emission_distribution = GaussianMixture(n_components=1)
            self.emission_distribution = [1,1,2,1]
            self.emission_distribution = normalized(self.emission_distribution)

        else:
            dim = len(self.substates)
            if dim > 0:
                self.transition_matrix = np.random.rand(dim, dim) # random transition matrix
                self.transition_matrix = sk.normalize(self.transition_matrix, norm='l1')
                self.transition_matrix[0] = 0
                self.initial_distribution = np.random.rand(dim) # random initial distribution
                self.initial_distribution = normalized(self.initial_distribution)

    def __str__(self, level=0):
        if (self.isProductionState):
            ret = "\t"*level+str(self.emission_distribution)+"\n"
        else:
            ret = "\t"*level+str(self.transition_matrix).replace("\n", "\n"+"\t"*level)+"\n"
            ret += "\t"*level+str(self.initial_distribution)+"\n"
        for child in self.substates:
            ret += child.__str__(level+1)
        return ret
    
"""
    def load_transition_matrix(file_path):
    def load_initial_distribution(file_path):
    def load_emission_distribution(file_path):

    def load(file_path):
"""
    
