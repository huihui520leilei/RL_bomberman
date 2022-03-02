import os
import random

import numpy as np
import pandas as pd


from .auxiliary import get_surrounding, get_goal_information_from_game_state
from .auxiliary import TASK2_ACTIONS
from .auxiliary import max_goal_distance



def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if os.path.isfile('saved/q_table.npy'):
        self.q_table = np.load('saved/q_table.npy')
    else:
        self.logger.info("Warning: No Q-table loaded!")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # todo Exploration vs exploitation
    
    # if random_num > epsilon:
    # choose action via exploitation
    # else:
    # choose action via exploration


    epsilon = 1.0 / (game_state['round']*game_state['step'])

    if random.random() < 0.01:#epsilon: #0.01: 
        return np.random.choice(TASK2_ACTIONS, p=[.2, .2, .2, .2, .2])

    #self.logger.debug("Querying q-table for action.")
    features = state_to_features(game_state)
    action_index = np.argmax(self.q_table[features])


    return TASK2_ACTIONS[action_index]





def state_to_features(game_state: dict) -> list:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return (0, 0, 0, 0, 0, 0, 0) # in the first step, the old_game_state is none

    else:
        (mode, goal, x_next_goal, y_next_goal) = get_goal_information_from_game_state(game_state)

        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        explosions = game_state['explosion_map']
        [up, down, right, left] = get_surrounding((x, y), arena, explosions) # == 1 if occupied


        features = (up, down, right, left, x_next_goal, y_next_goal, goal) #features are indexes for q-table
        return features
