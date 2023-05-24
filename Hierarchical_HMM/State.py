class State:
    def __init__(self, transition_matrix, initial_distribution, emission_distribution):
        self.transition_matrix = transition_matrix # only if internal state
        self.initial_distribution = initial_distribution # only if internal state
        self.emission_distribution = emission_distribution # only if production state
        self.substates = []
    
    def add_substate(self, new_substate):
        self.substates.append(new_substate)
    