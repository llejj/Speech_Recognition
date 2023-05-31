import numpy as np
import sklearn.preprocessing as sk

num_emissions = 13

np.random.seed(seed=10)

def normalized(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0: 
       return v
    return v / norm

class State:
    def __init__(self, isProductionState, parent=None, state_index=None, a=None, pi=None, e_means=None, e_covs=None):
        self.isProductionState = isProductionState
        self.parent = parent
        self.state_index = state_index

        self.a = None
        self.pi = None
        self.e_means = e_means if e_means is not None else normalized(np.random.rand(num_emissions))
        self.e_covs = e_covs if e_covs is not None else sk.normalize(np.random.rand(num_emissions, num_emissions))

        self.substates = []  # substates[0] = end state
    
    def add_substate(self, new_substate):
        self.substates.append(new_substate)        

    def randomize(self):
        if self.isProductionState:
            self.emission_means = np.random.rand(13)
            # self.emission_distribution = GaussianMixture(n_components=1)
            self.emission_distribution = [1,1,2,1]
            self.emission_distribution = normalized(self.emission_distribution)

        else:
            dim = len(self.substates)
            if dim > 0:
                self.a = np.random.rand(dim, dim) # random transition matrix
                self.a = sk.normalize(self.a, norm='l1')
                self.a[0] = 0
                self.pi = np.random.rand(dim) # random initial distribution
                self.pi = normalized(self.pi)

    def __str__(self, level=0):
        if (self.isProductionState):
            ret = "\t"*level+str(self.emission_distribution)+"\n"
        else:
            ret = "\t"*level+str(self.a).replace("\n", "\n"+"\t"*level)+"\n"
            ret += "\t"*level+str(self.pi)+"\n"
        for child in self.substates:
            ret += child.__str__(level+1)
        return ret
    
"""
    def load_transition_matrix(file_path):
    def load_initial_distribution(file_path):
    def load_emission_distribution(file_path):

    def load(file_path):
"""
    
