import numpy as np
import State

class HierarchicalHMM:
    def __init__(self, root):
        self.root = root

    

        

    # Generalized Baum-Welch
    def train(self, training_data):
        for level in range(self.num_levels):
            num_states = self.num_states_per_level[level]
            num_obs = np.max(training_data[level]) + 1  # Assumes 0-based indices

            # Initialize transition and emission matrices
            transition_matrix = np.zeros((num_states, num_states))
            emission_matrix = np.zeros((num_states, num_obs))

            # Compute transition and emission probabilities
            for sequence in training_data[level]:
                for i in range(1, len(sequence)):
                    current_state = sequence[i]
                    prev_state = sequence[i - 1]
                    transition_matrix[prev_state, current_state] += 1

                for state, obs in zip(sequence, training_data[0]):
                    emission_matrix[state, obs] += 1

            # Normalize transition and emission probabilities
            transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
            emission_matrix /= np.sum(emission_matrix, axis=1, keepdims=True)

            # Append matrices to the list
            self.transition_matrices.append(transition_matrix)
            self.emission_matrices.append(emission_matrix)

    # Generalized Viterbi
    def decode(self, observation_sequence):
        num_levels = len(self.transition_matrices)
        num_obs = len(observation_sequence)
        num_states_per_level = self.num_states_per_level

        # Initialize variables
        path = [0] * num_levels
        state_sequence = []

        # Forward-backward algorithm
        for t in range(num_obs):
            best_scores = np.zeros(num_states_per_level[num_levels - 1])
            best_paths = np.zeros(num_states_per_level[num_levels - 1], dtype=int)

            if t == 0:
                for state in range(num_states_per_level[num_levels - 1]):
                    best_scores[state] = np.log(self.emission_matrices[num_levels - 1][state, observation_sequence[t]])
            else:
                for state in range(num_states_per_level[num_levels - 1]):
                    scores = best_scores + np.log(self.transition_matrices[num_levels - 1][:, state])
                    best_paths[state] = np.argmax(scores)
                    best_scores[state] = scores[best_paths[state]] + np.log(self.emission_matrices[num_levels - 1][state, observation_sequence[t]])

            for level in range(num_levels - 2, -1, -1):
                best_scores_prev = best_scores
                best_paths_prev = best_paths
                best_scores = np.zeros(num_states_per_level[level])
                best_paths = np.zeros(num_states_per_level[level], dtype=int)

                for state in range(num_states_per_level[level]):
                    scores = best_scores_prev + np.log(self.transition_matrices[level][:, state])
                    best_paths[state] = np.argmax(scores)
                    best_scores[state] = scores[best_paths[state]] + np.log(self.emission_matrices[level][state, best_paths_prev])

            state_sequence.append(best_paths[0])
            for level in range(1, num_levels):
                path[level] = best_paths[path[level - 1]]

        return state_sequence[::-1]