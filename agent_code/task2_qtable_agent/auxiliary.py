import math

import pickle # debugging

init_rand_min = -2.0
init_rand_max = -2.0

### parameters
max_goal_distance = 4
learning_rate = 0.3
discount = 0.95

x_fov = max_goal_distance * 2 + 1 # number of possibilities for relative position of the nearest coin
y_fov = max_goal_distance * 2 + 1
number_of_goals = 2



### definitions
TASK2_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT'] # task 2 -> exclude 'WAIT' 
TASK2_QTABLE_SIZE = (3, 3, 3, 3, 3, x_fov, y_fov, number_of_goals, len(TASK2_ACTIONS)) # (mode, up, down, right, left, x_fov, y_fov, goals, n_actions) 


### functions

def get_crates_position_list(game_state):
	# todo: crates have to be extracted from game_stae['field'] == 1 as list of (int, int) coordinates like the bomb_xys and coins
	field = game_state['field']
	crate_list = []
	for x in range(field.shape[0]):
		for y in range(field.shape[1]):
			if field[x,y] == 1:
				crate_list.append((x,y))
	return crate_list

def get_explosion_position_list(game_state):
	explosion_map = game_state['explosion_map']
	explosion_list =[]
	for x in range(explosion_map.shape[0]):
		for y in range(explosion_map.shape[1]):
			if explosion_map[x,y] != 0:
				explosion_list.append((x,y))
	return explosion_list

def action_opposite(action):
	if action == "LEFT":
		return "RIGHT"
	if action == "RIGHT":
		return "LEFT"
	if action == "UP":
		return "DOWN"
	if action == "DOWN":
		return "UP"
	return ""

def get_goal_information_from_game_state(game_state):

	if game_state is None:
		return (0,0,0,0) # in the first step, the old_game_state is none
	else:
		# Gather information about the game state (next 11 lines of code taken from rule-based agent)
		arena = game_state['field']
		_, score, bombs_left, (x, y) = game_state['self']
		bombs = game_state['bombs']
		bomb_xys = [xy for (xy, t) in bombs]
		others = [xy for (n, s, b, xy) in game_state['others']]
		coins = game_state['coins']

		explosion_xys = get_explosion_position_list(game_state)
		crate_xys = get_crates_position_list(game_state) 

		mode = 0
		goal = 0
		[x_next_goal, y_next_goal] = [0,0]

		if len(explosion_xys): # explosions active
			mode = -1
			goal = 1 
			[x_next_goal, y_next_goal] = get_relative_position_to_nearest_goal((x, y), explosion_xys, max_goal_distance)
			return (mode, goal, x_next_goal, y_next_goal)
		elif len(bomb_xys) > 0: # bomb exists
				mode = -1 # escape
				goal = 0 # bomb
				[x_next_goal, y_next_goal] = get_relative_position_to_nearest_goal((x, y), bomb_xys, max_goal_distance)
				return (mode, goal, x_next_goal, y_next_goal)
		else:
			mode = 1 # approach goal
			if len(crate_xys) > 0: # crates exist (but no bombs)
				goal = 0 # crate
				[x_next_goal, y_next_goal] = get_relative_position_to_nearest_goal((x, y), crate_xys, max_goal_distance)
				return (mode, goal, x_next_goal, y_next_goal)
			else:
				if len(coins) > 0: # coins exist (but no bombs and no crates)
					goal = 1 # coins
					[x_next_goal, y_next_goal] = get_relative_position_to_nearest_goal((x, y), coins, max_goal_distance)
					return (mode, goal, x_next_goal, y_next_goal)
				else: # default values
					mode = 0
					goal = -1
					[x_next_goal, y_next_goal] = [0,0]
					return (mode, goal, x_next_goal, y_next_goal)




def get_relative_coin_positions(position, coins):
	"""Returns a list of relative coin positions.

	Keyword arguments:
    position -- current agent position from game_state
    coins -- the coins from game_state['coins']
    """
	relative_coin_positions = []
	for coin in coins:
		relative_position = (coin[0] - position[0], coin[1] - position[1])
		relative_coin_positions.append(relative_position)
	return relative_coin_positions


def get_relative_position_to_nearest_goal(position, coins, max_distance):
	"""Returns the relative coin position of the nearest GOAL by using the logic of get_relative_position_to_nearest_coin from task 1.

	Keyword arguments:
    position -- current agent position from game_state
    coins -- the coins from game_state['coins']
    max_distance -- the field of view for relative positions
    """
	relative_coin_positions = get_relative_coin_positions(position, coins)
	relative_position_to_nearest_coin = (0, 0)
	min_distance = 100
	for relative_coin_position in relative_coin_positions:
		distance = math.sqrt(relative_coin_position[0]**2 + relative_coin_position[1]**2)
		if distance < min_distance:
			relative_position_to_nearest_coin = relative_coin_position
			min_distance = distance
	delta_x = crop(relative_position_to_nearest_coin[0], max_distance)
	delta_y = crop(relative_position_to_nearest_coin[1], max_distance)
	return (delta_x, delta_y)

		
def crop(value, abs_max):
	if value > abs_max:
		return abs_max
	if value < -abs_max:
		return -abs_max
	return value


def get_surrounding(my_pos, arena, explosions): # order: up, down, right, left
	x = my_pos[0]
	y = my_pos[1]
	# coordinates
	xy_up = (x, y-1) 
	xy_down = (x, y+1)
	xy_right = (x+1, y)
	xy_left = (x-1, y)

	
	if explosions[xy_up] != 0: # a explosion is active
		up_val = 2
	else: # no explosion
		up_val = abs(arena[xy_up]) # arena gives (+1 for crates, -1 for stones/wall, 0  for free tiles)

	if explosions[xy_down] != 0: # a explosion is active
		down_val = 2
	else: # no explosion
		down_val = abs(arena[xy_down]) # arena gives (+1 for crates, -1 for stones/wall, 0  for free tiles)

	if explosions[xy_right] != 0: # a explosion is active
		right_val = 2
	else: # no explosion
		right_val = abs(arena[xy_right]) # arena gives (+1 for crates, -1 for stones/wall, 0  for free tiles)

	if explosions[xy_left] != 0: # a explosion is active
		left_val = 2
	else: # no explosion
		left_val = abs(arena[xy_left]) # arena gives (+1 for crates, -1 for stones/wall, 0  for free tiles)
	return [up_val, down_val, right_val, left_val]

def features_to_string(features):
	(up, down, right, left, x_next_coin, y_next_coin) = features
	my_2d_string_array =  [ ['', up, ''], [left, 'x', right], ['', down, ''] ]

	return my_2d_string_array
