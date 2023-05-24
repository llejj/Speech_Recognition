from State import State
from Hierarchical_HMM import Hierarchical_HMM
import numpy as np

state0 = State(False)

for i in range(2):
    new_state = State(False)
    for j in range(2):
        new_state2 = State(False, new_state, j)
        for k in range(2):
            new_state3 = State(True, new_state2, k)
            new_state2.add_substate(new_state3)
        new_state.add_substate(new_state2)
    state0.add_substate(new_state)

model = Hierarchical_HMM(state0)
model.randomize()


output_sequence = [0,0]
prob = model.output_sequence_prob(output_sequence)

print(prob)

(prob_state_sequence, q2last) = model.prob_q2last(output_sequence)
print(prob_state_sequence)
print(q2last)


print(str(state0))

model.update([1,2,3])
print(str(state0))
