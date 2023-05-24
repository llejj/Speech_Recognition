from State import State
from Hierarchical_HMM import Hierarchical_HMM
import numpy as np

state0 = State(False)

for i in range(5):
    new_state = State(True)
    new_state.randomize()
    state0.add_substate(new_state)

state0.randomize()

model = Hierarchical_HMM(state0)


output_sequence = [1,0,1,2,1,2,1,0]
prob = model.output_sequence_prob(output_sequence)

print(prob)
