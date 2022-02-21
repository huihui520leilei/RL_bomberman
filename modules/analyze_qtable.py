import os
import sys
import math
#import pickle
import numpy as np
import time


np.set_printoptions(threshold=sys.maxsize)

current_dir = os.getcwd()
os.chdir('/home/holger/git/ML_project/agent_code/number1a_agent')
#file = open("my-saved-q_table.pt", 'rb')
#q_table = pickle.load(file)
q_table = np.load('q_table.npy')
os.chdir(current_dir)

print(q_table.reshape(-1).shape) # flattened for size
q_is_minus_two = len(q_table[q_table==-2.0])
print(f"q_is_minus_two: {q_is_minus_two}")
q_is_zero = len(q_table[q_table==0.0])
print(f"q_is_zero: {q_is_zero}")
q_is_greater_than_zero = len(q_table[0<q_table])
print(f"q > 0: {q_is_greater_than_zero}")
print(f"maximum value: {np.max(q_table)}")
print(f"minimum value: {np.min(q_table)}")

#q_is_minus_inf = len(q_table[np.isinf(q_table)])
#print(f"q_is_minus_inf: {q_is_minus_inf}")


### coin collection states
#termination_states = np.argwhere(q_table==-2.0)
#for termination_state in termination_states:
	#print(termination_state)

import matplotlib.pyplot as plt
q_flat=q_table.reshape(-1)
_ = plt.hist(q_flat, bins='auto')
plt.show(block=False)
plt.pause(3)
plt.close()
