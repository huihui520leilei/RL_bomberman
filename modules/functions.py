import math

def get_relative_coin_positions(position, coins):
	"""Returns the relative coin positions."""
	relative_coin_positions = []
	for coin in coins:
		relative_position = (coin[0] - position[0], coin[1] - position[1])
		relative_coin_positions.append(relative_position)
	return relative_coin_positions


def get_relative_coin_positions_and_nearest_coin_index(position, coins):
	"""Returns the relative coin positions and the index of the nearest coin. Index is -1, if coin list is empty."""
	relative_coin_positions = []
	nearest_coin_index = -1
	min_distance = 100
	index = 0
	for coin in coins:
		relative_position = (coin[0] - position[0], coin[1] - position[1])
		relative_coin_positions.append(relative_position)
		distance = math.sqrt(relative_position[0]**2 + relative_position[1]**2)
		if distance < min_distance:
			nearest_coin_index = index
			min_distance = distance
		index = index+1
	return relative_coin_positions, nearest_coin_index


def get_relative_position_to_nearest_coin(position, coins, max_distance): 
	relative_coin_positions = get_relative_coin_positions(position, coins)
	relative_position_to_nearest_coin = (0,0)
	min_distance = 100
	for relative_coin_position in relative_coin_positions:
		distance = math.sqrt(relative_coin_position[0]**2 + relative_coin_position[1]**2)
		if distance < min_distance:
			relative_position_to_nearest_coin = relative_coin_position
			min_distance = distance
	# add offset for positive values only
	delta_x = relative_position_to_nearest_coin[0] 
	delta_y = relative_position_to_nearest_coin[1] 
	#print(f'before cropping: {(delta_x, delta_y)}')
	delta_x = crop(delta_x, max_distance)
	delta_y = crop(delta_y, max_distance)
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
	xy_up = (x-1, y)
	xy_down = (x+1, y)
	xy_right = (x, y+1)
	xy_left = (x, y-1)
	return [abs(arena[xy_up]), abs(arena[xy_down]), abs(arena[xy_right]), abs(arena[xy_left])]
