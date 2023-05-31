import numpy as np
from State_continuous import State_continuous

class Hierarchical_HMM:
    def __init__(self, root):
        self.root = root

    # Calculating the likelihood of a sequence
    def alpha(self, t_start, t_end, i, state, output_sequence):
        substate = state.substates[i]
        if (substate.isProductionState):
            if (t_start == t_end):
                return (state.pi[i]) * substate.emission_distribution[output_sequence[t_start - 1]]
            else:
                answer = 0
                for j in range(len(state.substates)):
                    answer += self.alpha(t_start, t_end - 1, j, state, output_sequence) * state.a[j][i]
                answer *= substate.emission_distribution[output_sequence[t_end - 1]]
                return answer
        else:
            if (t_start == t_end):
                answer = 0
                for s in range(len(substate.substates)):
                    answer += self.alpha(t_start, t_start, s, substate, output_sequence) * substate.a[s][0]
                answer *= state.pi[i]
                return answer
            else:
                answer = 0
                for l in range(t_end - t_start):
                    answer1 = 0
                    for j in range(len(state.substates)):
                        answer1 += self.alpha(t_start, t_start + l, i, state, output_sequence) * state.a[j][i]
                    answer2 = 0
                    for s in range(len(substate.substates)):
                        answer2 += self.alpha(t_start + l + 1, t_end, s, substate, output_sequence) * substate.a[s][0]
                    answer += answer1 * answer2

                answer3 = 0
                for s in range(len(substate.substates)):
                    answer3 += self.alpha(t_start, t_end, s, substate, output_sequence) * substate.a[s][0]
                answer *= state.pi[i]

                answer += answer3
                return answer
    def beta(self, t_start, t_end, i, state, output_sequence):
        substate = state.substates[i]
        if (substate.isProductionState):
            if (t_start == t_end):
                return substate.emission_distribution[output_sequence[t_start - 1]] * state.a[i][0]
            else:
                answer = 0
                for j in range(1, len(state.substates)):
                    answer += self.beta(t_start + 1, t_end, j, state, output_sequence) * state.a[i][j]
                answer *= substate.emission_distribution[output_sequence[t_start - 1]]
                return answer
        else:
            if (t_start == t_end):
                answer = 0
                for s in range(len(substate.substates)):
                    answer += self.beta(t_start, t_start, s, substate, output_sequence) * substate.pi[s]
                answer *= state.a[i][0]
                return answer
            else:
                answer = 0
                for l in range(t_end - t_start):
                    answer1 = 0
                    for s in range(len(substate.substates)):
                        answer1 += self.beta(t_start, t_start + l, s, substate, output_sequence) * substate.pi[s]
                    answer2 = 0
                    for j in range(len(state.substates)):
                        answer2 += self.beta(t_start + l + 1, t_end, j, state, output_sequence) * state.a[i][j]
                    answer += answer1 * answer2

                answer3 = 0
                for s in range(len(substate.substates)):
                    answer3 += self.beta(t_start, t_end, s, substate, output_sequence) * substate.pi[s]
                answer *= state.a[i][0]

                answer += answer3
                return answer

    # forward-backward
    def output_sequence_prob(self, output_sequence):
        answer = 0
        for i in range(len(self.root.substates)):
            answer += self.alpha(1, len(output_sequence), i, self.root, output_sequence)
        return answer

    # Viterbi
    def delta_psi_tau(self, t_start, t_end, i, state, output_sequence):
        substate = state.substates[i]
        if (substate.isProductionState):
            if (t_start == t_end):
                delta = (state.pi[i]) * substate.emission_distribution[output_sequence[t_start - 1]]
                return (delta, 0, t_start)
            else:
                delta = 0 # 0 is lower bound for delta
                psi = 0
                for j in range(len(substate.substates)):
                    if (delta < self.delta_psi_tau(t_start, t_end - 1, j, state, output_sequence)[0] * state.a[j][i] * substate.emission_distribution[t_end - 1]):
                        delta = self.delta_psi_tau(t_start, t_end - 1, j, state, output_sequence)[0] * state.a[j][i] * substate.emission_distribution[t_end - 1]
                        psi = j
                return (delta, psi, t_end)
        else:
            if (t_start == t_end):
                delta = 0                
                for j in range(len(substate.substates)):
                    if (delta < state.pi[i] * self.delta_psi_tau(t_start, t_start, j, substate, output_sequence)[0] * substate.a[j][0]):
                        delta = state.pi[i] * self.delta_psi_tau(t_start, t_start, j, substate, output_sequence)[0] * substate.a[j][0]
                
                return (delta, 0, t_start)
            else:
                # t' == t_start:
                delta_t = 0
                for r in range(len(substate.substates)):
                    if (delta_t < self.delta_psi_tau(t_start, t_end, r, substate, output_sequence)[0] * substate.a[r][0]):
                        delta_t = self.delta_psi_tau(t_start, t_end, r, substate, output_sequence)[0] * substate.a[r][0]
                delta_t *= state.pi[i]
                psi = 0
                tau = t_start

                # t' > t_start:
                for t_prime in range(t_start + 1, t_end):

                    # find R
                    _R = 0
                    for r in range(len(substate.substates)):
                        if (_R < self.delta_psi_tau(t_prime, t_end, r, substate, output_sequence)[0] * substate.a[r][0]):
                            _R = self.delta_psi_tau(t_prime, t_end, r, substate, output_sequence)[0] * substate.a[r][0]
                    # find new_delta_t
                    new_delta_t = 0
                    new_psi = 0
                    for j in range(len(state.substates)):
                        if (new_delta_t < self.delta_psi_tau(t_start, t_prime - 1, j, state, output_sequence)[0] * state.a[j][i] * _R):
                            new_delta_t = self.delta_psi_tau(t_start, t_prime - 1, j, state, output_sequence)[0] * state.a[j][i] * _R
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



    