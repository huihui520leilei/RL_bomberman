import math
import numpy as np

### parameters
max_coin_distance = 2
x_fov = max_coin_distance * 2 + 1 # number of possibilities for relative position of the nearest coin
y_fov = max_coin_distance * 2 + 1

### definitions
TASK1_ACTION_TABLE_SIZE = (2, 2, 2, 2, x_fov, y_fov)
#TASK1_ACTION_TABLE_SIZE = (2, 2, 2, 2, x_fov, y_fov, 6) # (up, down, right, left, x_fov, y_fov) 

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
	xy_up = (x-1, y)
	xy_down = (x+1, y)
	xy_right = (x, y+1)
	xy_left = (x, y-1)
	return [abs(arena[xy_up]), abs(arena[xy_down]), abs(arena[xy_right]), abs(arena[xy_left])]

def look_for_targets_coin_collector(free_space, start, targets, logger=None):
    """Find direction of the closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards the closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        #shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    return current

def relative_position_to_target(target, position):
    relative_position = (target[0] - position[0], target[1] - position[1])
    return relative_position