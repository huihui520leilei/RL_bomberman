import math
import numpy as np
import random




x = np.array([[3, 0, 0, 1], [0, 4, 0, 0], [5, 6, 0, 0]])
y = np.array([[3,1], [0,2], [5,  0]])
z = np.nonzero(np.any(x != 0, axis=1))[0]
print(f"z: {z}")
print(f"x: {x}")
print(f"x.shape: {x.shape}")
print(f"y.shape: {y.shape}")
zz = y[z]
print(zz)

for i in range(10):
	a = random.random()
	print(a)


import sys
sys.path.append("/home/holger/git/ML_project/modules")
import functions as f

epsilon = 0.5
nA = 6

A = np.ones(nA, dtype=float) * epsilon / nA

print(f"A: {A}")
best_action = 2
A[best_action] += (1.0 - epsilon)
print(f"A: {A}")


b = np.array([[ 1, 2, 3, 4, 5, 6]])
print(b)
b[0][1] = 100
print(b)
 
  

a = np.array([[0,1,2,3,4,5,6,7,8,9]])
a = a.transpose()
print(f"a.shape: {a.shape}")

channels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
stacked_channels = np.stack(channels)
new = stacked_channels.reshape(10)
new = np.transpose(new)
print(f"shape of new: {new.shape}")


#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
#action = ACTIONS.index(self_action)




X = [[-1, 0, 0, -1, 2, 1, 0, 0, 0, 0], [-1, 0, 0, -1, 2, 1, 0, 0, 0, 0], [-1, 0, 0, -1, 2, 1, 0, 0, 0, 0]]
targets = [np.array([[ 0., -1.,  0.,  0.,  0.,  0.]]), np.array([[-1.,  0.,  0.,  0.,  0.,  0.]]), np.array([[ 0., -1.,  0.,  0.,  0.,  0.]])]
X= np.array(X)
targets = np.concatenate(targets, axis=0 )
print(X.shape)
print(targets.shape)


a = (1.0, 2.0)
b = (4, 9)

channels = []
channels.append(a)
channels.append(b)
# concatenate them as a feature tensor (they must have the same shape), ...
stacked_channels = np.stack(channels)
# and return them as a vector
print(stacked_channels.reshape(-1))





# load q-table
q_table = np.load('../agent_code/number1a_agent/q_table.npy')
#file = open("../agent_code/number1a_agent/my-saved-q_table.pt", 'rb')
#q_table = pickle.load(file)
print(q_table.shape)
q_bigger_zero = len(q_table[q_table>0])
q_smaller_zero = len(q_table[q_table<0])
q_is_minus_two = len(q_table[q_table==-2.0])
print(f"maximum value: {np.max(q_table)}")
print(f"q_is_minus_two: {q_is_minus_two}")
print(f"q_bigger_zero: {q_bigger_zero}")
print(f"q_smaller_zero: {q_smaller_zero}")
print(f"sum: {q_smaller_zero+q_bigger_zero}")

print(q_table.reshape(-1).shape) # flattened for size



# check arena
arena = np.zeros((17, 17)).astype(int)
arena[:1, :] = -1
arena[-1:, :] = -1
arena[:, :1] = -1
arena[:, -1:] = -1
for x in range(17):
	for y in range(17):
		if (x + 1) * (y + 1) % 2 == 1:
			arena[x, y] = -1
arena[5,2] = 3
arena[5,0] = 8
print(arena)
my_pos = (5,4)
my = f.get_surrounding(my_pos, arena)
print('up, down, right, left')
print(my)







position = (15, 15)
coins = [(4, 9), (5, 11), (10, 5), (10, 7), (6, 15), (15, 1)]

# get first coin
coin = coins[0]


relative_position_to_nearest_coin = f.get_relative_position_to_nearest_coin(position, coins,10)
print(f"positions: {position}")
print(f"coins: {coins}")
print(f"relative_position_to_nearest_coin: {relative_position_to_nearest_coin}")





a = np.random.uniform(low=-1, high=0, size=(2,3))
a[:,1] = 1
print(a)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # , 'WAIT']#, 'BOMB']
b=np.random.choice(ACTIONS, p=[.25, .25, .25, .25], size=len(ACTIONS))#, .2])  # , .0])
print(b)

#q_table = np.random.uniform(low=-1, high=0, size=(5, 5, 5, 5, 4))
#self.q_table (:, :, :, :, )

print(np.argmax(np.array([-1, 1, 2, 3, 2])))

for i in range(10):
	print(random.random())


#my position (x, y): (9, 12)
#coins: [(3, 5), (2, 9), (3, 15), (7, 1), (8, 9)]
#relative coints: [(-6, -7), (-7, -3), (-6, 3), (-2, -11), (-1, -3)]
#next coin: (-1, -2)
(x_next_coin1, y_next_coin1) = f.get_relative_position_to_nearest_coin((9, 12), [(3, 5), (2, 9), (3, 15), (7, 1), (8, 9)], 2)
print((x_next_coin1, y_next_coin1))



#my position (x, y): (9, 13)
#coins: [(3, 5), (2, 9), (3, 15), (7, 1), (8, 9)]
#relative coints: [(-6, -8), (-7, -4), (-6, 2), (-2, -12), (-1, -4)]
#next coin: (-1, -2)
(x_next_coin2, y_next_coin2) = f.get_relative_position_to_nearest_coin((9, 13), [(3, 5), (2, 9), (3, 15), (7, 1), (8, 9)], 2)
print((x_next_coin2, y_next_coin2))

