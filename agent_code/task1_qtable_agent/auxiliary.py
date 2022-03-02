import math

init_rand_min = -5.0
init_rand_max = -2.0

### parameters
max_coin_distance = 5
learning_rate = 0.5
discount = 0.95

x_fov = max_coin_distance * 2 + 1 # number of possibilities for relative position of the nearest coin
y_fov = max_coin_distance * 2 + 1


### definitions
TASK1_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT'] # task 1 -> exclude 'WAIT', 'BOMB'
TASK1_QTABLE_SIZE = (2, 2, 2, 2, x_fov, y_fov, len(TASK1_ACTIONS)) # (up, down, right, left, x_fov, y_fov, n_actions) 


### functions
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


def get_relative_position_to_nearest_coin(position, coins, max_distance):
	"""Returns the relative coin position of the nearest coin.

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


def get_surrounding(my_pos, arena): # order: up, down, right, left
	x = my_pos[0]
	y = my_pos[1]
	xy_up = (x, y-1)
	xy_down = (x, y+1)
	xy_right = (x+1, y)
	xy_left = (x-1, y)
	return [abs(arena[xy_up]), abs(arena[xy_down]), abs(arena[xy_right]), abs(arena[xy_left])]

def features_to_string(features):
	(up, down, right, left, x_next_coin, y_next_coin) = features
	my_2d_string_array =  [ ['', up, ''], [left, 'x', right], ['', down, ''] ]

	return my_2d_string_array
