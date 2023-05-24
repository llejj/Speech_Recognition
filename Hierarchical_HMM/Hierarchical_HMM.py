import numpy as np
from State import State

class Hierarchical_HMM:
    def __init__(self, root):
        self.root = root

    # Calculating the likelihood of a sequence
    def alpha(self, t_start, t_end, i, state, output_sequence):
        substate = state.substates[i]
        if (substate.isProductionState):
            if (t_start == t_end):
                return (state.initial_distribution[i]) * substate.emission_distribution[output_sequence[t_start - 1]]
            else:
                answer = 0
                for j in range(len(state.substates)):
                    answer += self.alpha(t_start, t_end - 1, j, state, output_sequence) * state.transition_matrix[j][i]
                answer *= substate.emission_distribution[output_sequence[t_end - 1]]
                return answer
        else:
            if (t_start == t_end):
                answer = 0
                for s in range(len(substate.substates)):
                    answer += self.alpha(t_start, t_start, s, substate, output_sequence) * substate.transition_matrix[s][0]
                answer *= state.initial_distribution[i]
                return answer
            else:
                answer = 0
                for l in range(t_end - t_start):
                    answer1 = 0
                    for j in range(len(state.substates)):
                        answer1 += self.alpha(t_start, t_start + l, i, state, output_sequence) * state.transition_matrix[j][i]
                    answer2 = 0
                    for s in range(len(substate.substates)):
                        answer2 += self.alpha(t_start + l + 1, t_end, s, substate, output_sequence) * substate.transition_matrix[s][0]
                    answer += answer1 * answer2

                answer3 = 0
                for s in range(len(substate.substates)):
                    answer3 += self.alpha(t_start, t_end, s, substate, output_sequence) * substate.transition_matrix[s][0]
                answer *= state.initial_distribution[i]

                answer += answer3
                return answer
    def beta(self, t_start, t_end, i, state, output_sequence):
        substate = state.substates[i]
        if (substate.isProductionState):
            if (t_start == t_end):
                return substate.emission_distribution[output_sequence[t_start - 1]] * state.transition_matrix[i][0]
            else:
                answer = 0
                for j in range(1, len(state.substates)):
                    answer += self.beta(t_start + 1, t_end, j, state, output_sequence) * state.transition_matrix[i][j]
                answer *= substate.emission_distribution[output_sequence[t_start - 1]]
                return answer
        else:
            if (t_start == t_end):
                answer = 0
                for s in range(len(substate.substates)):
                    answer += self.beta(t_start, t_start, s, substate, output_sequence) * substate.initial_distribution[s]
                answer *= state.transition_matrix[i][0]
                return answer
            else:
                answer = 0
                for l in range(t_end - t_start):
                    answer1 = 0
                    for s in range(len(substate.substates)):
                        answer1 += self.beta(t_start, t_start + l, s, substate, output_sequence) * substate.initial_distribution[s]
                    answer2 = 0
                    for j in range(len(state.substates)):
                        answer2 += self.beta(t_start + l + 1, t_end, j, state, output_sequence) * state.transition_matrix[i][j]
                    answer += answer1 * answer2

                answer3 = 0
                for s in range(len(substate.substates)):
                    answer3 += self.beta(t_start, t_end, s, substate, output_sequence) * substate.initial_distribution[s]
                answer *= state.transition_matrix[i][0]

                answer += answer3
                return answer

    # forward-backward
    def output_sequence_prob(self, output_sequence):
        answer = 0
        for i in range(len(self.root.substates)):
            answer += self.alpha(1, len(output_sequence), i, self.root, output_sequence)
        return answer
    
    # Baum-Welch (EM)
    def eta_in(self, t, i, state, output_sequence):
        if (state.parent == None):
            if (t == 1):
                return state.initial_distribution(i)
            elif (t > 1):
                answer = 0
                for j in range(len(state.substates)):
                    answer += self.alpha(self, 1, t - 1, j, state, output_sequence) * state.transition_matrix[j][i]
                return answer
            else:
                print("eta_in error")
        else:
            if (t == 1):
                return self.eta_in(1, state.state_index, state.parent, output_sequence) * state.initial_distribution[i]
            elif (t > 1):
                answer1 = 0
                for t_prime in range(t - 1):
                    answer2 = 0
                    for j in range(len(state.substates)):
                        answer2 += self.alpha(t_prime, t - 1, j, state, output_sequence) * state.transition_matrix[j][i]

                    answer1 += self.eta_in(t_prime, state.state_index, state.parent, output_sequence) * answer2
                answer += self.eta_in(t, state.state_index, state.parent, output_sequence) * state.initial_distribution[i]
            else:
                print("eta_in error")

    def eta_out(self, t, i, state, output_sequence):
        if (t < len(output_sequence)):
            if (state.parent == None):
                answer = 0
                for j in range(len(state.substates)):
                    answer += state.transition_matrix[i][j] * self.beta(t + 1, len(output_sequence), j, state, output_sequence)
                return answer
            else:
                answer = 0
                for k in range(t+1, len(output_sequence)):
                    answer2 = 0
                    for j in range(len(state.substates)):
                        answer2 += state.transition_matrix[i][j] * self.beta(t+1, k, j, state.state_index, output_sequence)
                    answer2 *= self.eta_out(k, state.state_index, state.parent, output_sequence)
                    answer += answer2
                answer += state.transition_matrix[i][0] * self.eta_out(len(output_sequence), state.state_index, state.parent, output_sequence)
                return answer
        else:
            return state.transition_matrix[i][0] * self.eta_out(len(output_sequence), state.state_index, state.parent, output_sequence)
    
    def xi(self, t, i, j, state, output_sequence):
        if (state.parent == None):
            if (t < len(output_sequence)):
                numerator = self.alpha(1, t, i, state, output_sequence) * state.transition_martrix[i][j] * self.beta(t+1, len(output_sequence), j, state, output_sequence)
                denominator = self.output_sequence_prob(output_sequence)
                return numerator / denominator
            else:
                return self.alpha(1, len(output_sequence), i, state, output_sequence) * state.transition_matrix[i][j] / self.output_sequence_prob(output_sequence)
        elif (t < len(output_sequence)):
            if (not (j == 0)):
                answer1 = 0
                for s in range(t):
                    answer1 += self.eta_in(s, state.state_index, state.parent, output_sequence) * self.alpha(s, t, i, state, output_sequence)
                answer2 = 0
                for e in range(t+1, len(output_sequence)):
                    answer2 += self.beta(t+1, e, j, state, output_sequence) * self.eta_out(e, state.state_index, state.parent, output_sequence)
                return  answer1 * state.transition_matrix[i][j] * answer2 / self.output_sequence_prob(output_sequence)
            else:
                answer = 0
                for s in range(t):
                    answer += self.eta_in(s, state.state_index, state.parent, output_sequence) * self.alpha(s, t, i, state, output_sequence)
                answer *= state.transition_matrix[i][0] * self.eta_out(t, state.state_index, state.parent, output_sequence)
                answer /= self.output_sequence_prob(output_sequence)
                return answer
    
    def chi(self, t, i, state, output_sequence):
        if (state.parent == None):
            return state.initial_distribution[i] * self.beta(1, len(output_sequence), i, state, output_sequence) / self.output_sequence_prob(output_sequence)
        else:
            answer = 0
            for e in range(t, len(output_sequence)):
                answer += self.beta(t, e, i, state, output_sequence) * self.eta_out(e, state.state_index, state.parent, output_sequence)
            answer *= self.eta_in(t, state.state_index, state.parent, output_sequence) * state.initial_distribution(i) / self.output_sequence_prob(output_sequence)
    
    def gamma_in(self, t, i, state, output_sequence):
        answer = 0
        for k in range(len(state.substates)):
            answer += self.xi(t-1, k, i, state, output_sequence)
        return answer
    def gamma_out(self, t, i, state, output_sequence):
        answer = 0
        for k in range(len(state.substates)):
            answer += self.xi(t, i, k, state, output_sequence)
    
    def _update_initial_distribution(self, state, output_sequence):
        if (state.parent == None):
            for i in range(len(state.substates)):
                answer = 0
                for t in range(len(output_sequence)):
                    answer += self.chi(t, i, state, output_sequence)
                state.initial_distribution[i] = answer
        elif (not state.isProductionState):
            for i in range(len(state.substates)):
                numerator = 0
                for t in range(len(output_sequence)):
                    numerator += self.chi(t, i, state, output_sequence)
                denominator = 0
                for k in range(len(state.substates)):
                    for t in range(len(output_sequence)):
                        denominator += self.xi(t, i, k, state, output_sequence)
                state.initial_distribution[i] = numerator / denominator
    
    def _update_transition_matrix(self, state, output_sequence):
        for i in range(len(state.substates)):
            for j in range(len(state.substates)):
                numerator = 0
                denominator = 0
                for t in range(len(output_sequence)):
                    numerator += self.xi(t, i, j, state, output_sequence)
                    denominator +=  self.gamma_out(t, i, state, output_sequence)
                state.transition_matrix[i][j] = numerator / denominator
    
    def _update_emission_distribution(self, state, output_sequence):
        for v in range(len(state.emission_distribution)):
            if (state.substates[0].isProductionState):
                for i in range(len(state.substates)):
                    answer1 = 0
                    for t in range(len(output_sequence)):
                        if (output_sequence[t] == v):
                            answer1 += self.chi(t, i, state, output_sequence)
                    answer2 = 0
                    for t in range(1, len(output_sequence)):
                        if (output_sequence[t] == v):
                            answer2 += self.gamma_in(t, i, state, output_sequence)
                    
                    answer3 = 0
                    for t in range(len(output_sequence)):
                        answer3 += self.chi(t, i, state, output_sequence)
                    answer4 = 0
                    for t in range(1, len(output_sequence)):
                        answer4 += self.gamma_in(t, i, state, output_sequence)
                    state.substates[i].emission_distribution[v] = (answer1 + answer2) / (answer3 + answer4)
                    
    def _update(self, state, output_sequence):
        if (state.isProductionState):
            self._update_emission_distribution(state, output_sequence)
        else:
            self._update_initial_distribution(state, output_sequence)
            self._update_transition_matrix(state, output_sequence)
            for i in range(len(state.substates)):
                self._update(self, state.substates[i], output_sequence)
    
    def update(self, output_sequence):
        self._update(self.root, output_sequence)

        






    # Viterbi
    def delta_psi_tau(self, t_start, t_end, i, state, output_sequence):
        substate = state.substates[i]
        if (substate.isProductionState):
            if (t_start == t_end):
                delta = (state.initial_distribution[i]) * substate.emission_distribution[output_sequence[t_start - 1]]
                return (delta, 0, t_start)
            else:
                delta = 0 # 0 is lower bound for delta
                psi = 0
                for j in range(len(substate.substates)):
                    if (delta < self.delta_psi_tau(t_start, t_end - 1, j, state, output_sequence)[0] * state.transition_matrix[j][i] * substate.emission_distribution[t_end - 1]):
                        delta = self.delta_psi_tau(t_start, t_end - 1, j, state, output_sequence)[0] * state.transition_matrix[j][i] * substate.emission_distribution[t_end - 1]
                        psi = j
                return (delta, psi, t_end)
        else:
            if (t_start == t_end):
                delta = 0                
                for j in range(len(substate.substates)):
                    if (delta < state.initial_distribution[i] * self.delta_psi_tau(t_start, t_start, j, substate, output_sequence)[0] * substate.transition_matrix[j][0]):
                        delta = state.initial_distribution[i] * self.delta_psi_tau(t_start, t_start, j, substate, output_sequence)[0] * substate.transition_matrix[j][0]
                
                return (delta, 0, t_start)
            else:
                # t' == t_start:
                delta_t = 0
                for r in range(len(substate.substates)):
                    if (delta_t < self.delta_psi_tau(t_start, t_end, r, substate, output_sequence)[0] * substate.transition_matrix[r][0]):
                        delta_t = self.delta_psi_tau(t_start, t_end, r, substate, output_sequence)[0] * substate.transition_matrix[r][0]
                delta_t *= state.initial_distribution[i]
                psi = 0
                tau = t_start

                # t' > t_start:
                for t_prime in range(t_start + 1, t_end):

                    # find R
                    _R = 0
                    for r in range(len(substate.substates)):
                        if (_R < self.delta_psi_tau(t_prime, t_end, r, substate, output_sequence)[0] * substate.transition_matrix[r][0]):
                            _R = self.delta_psi_tau(t_prime, t_end, r, substate, output_sequence)[0] * substate.transition_matrix[r][0]
                    # find new_delta_t
                    new_delta_t = 0
                    new_psi = 0
                    for j in range(len(state.substates)):
                        if (new_delta_t < self.delta_psi_tau(t_start, t_prime - 1, j, state, output_sequence)[0] * state.transition_matrix[j][i] * _R):
                            new_delta_t = self.delta_psi_tau(t_start, t_prime - 1, j, state, output_sequence)[0] * state.transition_matrix[j][i] * _R
                            new_psi = j
                    
                    if (delta_t < new_delta_t):
                        delta_t = new_delta_t
                        psi = new_psi
                        tau = t_prime
                return (delta_t, psi, tau)
    def prob_q2last(self, output_sequence):
        prob = 0
        q2last = 0
        for i in range(len(self.root.substates)):
            if (prob < self.delta_psi_tau(1, len(output_sequence), i, self.root, output_sequence)[0]):
                prob = self.delta_psi_tau(1, len(output_sequence), i, self.root, output_sequence)[0]
                q2last = i
        return (prob, q2last)


    # extras
    def _randomize(self, state):
        state.randomize()
        if (not state.isProductionState):
            for i in range(0, len(state.substates)):
                self._randomize(state.substates[i])
    
    def randomize(self):
        self._randomize(self.root)



    