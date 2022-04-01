import os
import pickle
import random
import numpy as np
import pandas as pd
from random import shuffle
from .RuleBased import act as rule_act
from .RuleBased import setup as rule_setup
from .RuleBased import look_for_targets


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']#, 'WAIT', 'BOMB']

#-------------------------------------------------------------------------------------------------------
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
    rule_setup(self)
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    #-------------------------------------------------------
    if os.path.isfile('q_table.pickle'):
        with open('q_table.pickle', 'rb') as file:
            self.q_table = pickle.load(file)
#-------------------------------------------------------------------------------------------------------   
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    ##Gather information about the game state-------------------------------
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    ##-----------------------------------------------------------------------
    #Find coordinate of closest targets
    targets = coins
    nearestCoins = targets[np.argmin(np.sum(np.abs(np.subtract(targets, (x,y))), axis=1))]
    [x_nearest_coin, y_nearest_coin] = nearestCoins
    #==============================Print=======================================
    #print("***********************************************")
    # print("------------")
    # print(f'Current agent position: {(x,y)}')
    # print(f'All coins position:     {targets}')
    # print(f'Nearest coin position:  {nearestCoins}')
    # print("------------")
    #=========================================================================

    if self.train:
    #if training, Use Rule based agent for tranining#####################################
        if game_state is not None:
            Action = rule_act(self, game_state)
        else:
            Action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
        return Action
    #################################################################################   
    else:
        ##otherwise, choose action based on Q table, if there have more action with same Q value, random choose in those action
        if (x,y,x_nearest_coin, y_nearest_coin) not in self.q_table.index:
            Action = rule_act(self, game_state)
        else:
            stateActions = self.q_table.loc[[(x,y,x_nearest_coin, y_nearest_coin)]]
            MaxQval_one = stateActions.max(1).tolist()*len(ACTIONS)# in this state, the MaxValue action in Q table
            MaxQval_multi = np.where(stateActions == MaxQval_one)[1]# in this state, the action with same MaxValue in Q table
            Actions_all = np.array(ACTIONS)[MaxQval_multi.astype(int)]
            Action = np.random.choice(Actions_all)
            #print("222222222222222222222222222222222222222222")
        #todo Exploration vs exploitation-----------------------------------
        random_prob = .01
        if self.train and random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
        self.logger.debug("Querying model for action.")
        return Action


#--------------------------------------------------------------------------------------------------------------    
def state_to_features(game_state: dict) -> np.array:
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
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
