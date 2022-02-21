import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import random
import numpy as np
import math
import pandas as pd
from random import shuffle
from .RuleBased import act as rule_act
from .RuleBased import setup as rule_setup
from .RuleBased import look_for_targets
from tensorflow import keras
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
            self.n_actions = len(ACTIONS)
            self.n_features = 16
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
            self.DQNpara = pickle.load(file)
            weight_L1 = np.array(self.DQNpara[0],dtype=object)
            bias_L1 = np.array(self.DQNpara[1],dtype=object)
            weight_L2 = np.array(self.DQNpara[2],dtype=object)
            bias_L2 = np.array(self.DQNpara[3],dtype=object)    
            w1_initializer, b1_initializer = tf.constant_initializer(weight_L1), tf.constant_initializer(bias_L1)
            w2_initializer, b2_initializer = tf.constant_initializer(weight_L2), tf.constant_initializer(bias_L2)
            
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w1_initializer,
                                     bias_initializer=b1_initializer, name='e1')
                self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w2_initializer,
                                              bias_initializer=b2_initializer, name='q')
            self.sess = tf.Session() 
            self.sess.run(tf.global_variables_initializer())  
#-------------------------------------------------------------------------------------------------------   
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    Observation = state_to_features(game_state)
    #--------------------------------------------------------------------------------
    if self.train:
    #if training, Use Rule based agent for tranining#################################
        if game_state is not None:
            Action = rule_act(self, game_state)
        else:
            Action = np.random.choice(ACTIONS, p=[.20, .20, .20, .20, 0.1, 0.1])
            
        return Action
    #################################################################################   
    else:
        ##otherwise, choose action based on Deep_Q_Network
        observation = state_to_features(game_state)
        observation = observation[np.newaxis, :]
        
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)        
        Action = ACTIONS[action]
        print("**********************")
        print(Action)
        #todo Exploration vs exploitation---------------------------------------------
        random_prob = .000000001
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.20, .20, .20, .20, 0.1, 0.1])
        self.logger.debug("Querying model for action.")
        return Action
######################################################################################
def get_relative_positions(position, targets):
    """Returns the relative coin positions."""
    position_x = position[0]
    position_y = position[1]
    relative_positions = []
    if not targets:
        return relative_positions
    for target in targets:
        target_x = target[0]
        target_y = target[1]
        relative_position = (target_x - position_x, target_y - position_y)
        relative_positions.append(relative_position)
    return relative_positions
#-------------------------------------------------------
def get_relative_position_to_nearest(position, targets, max_distance=100):
    relative_positions = get_relative_positions(position, targets)
    relative_position_to_nearest = (0, 0)
    if not relative_positions:
        return relative_position_to_nearest
    min_distance = 100
    for relative_position in relative_positions:
        distance = math.sqrt(relative_position[0] ** 2 + relative_position[1] ** 2)
        if distance < min_distance:
            relative_position_to_nearest = relative_position
            min_distance = distance
    # add offset for positive values only
    delta_x = relative_position_to_nearest[0]
    delta_y = relative_position_to_nearest[1]
    return (delta_x, delta_y)
#---------------------------------------------------------
# def get_relative_position_to_nearest_bomb(position, bombs, max_distance=100):
    # if not bombs:
        # return (0,0)
    # else:
        # bombs = []
        # for bomb in bombs:
            # bombs.append(bomb[0])
    # return get_relative_position_to_nearest(position, bombs, max_distance=100)
#########################################################################################   
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
        arena_0 = 0
        arena_1 = 0
        arena_2 = 0
        arena_3 = 0
        arena_4 = 0
        explosion_0 = 0
        explosion_1 = 0
        explosion_2 = 0
        explosion_3 = 0
        explosion_4 = 0
        coins = (0,0)
        crates = (0,0)
        dead_ends = (0,0)
        x_nearest_Coin = 0
        y_nearest_Coin = 0
        x_nearest_Crates = 0
        y_nearest_Crates = 0
        x_nearest_Dead_ends = 0
        y_nearest_Dead_ends = 0
    else:
        ##Gather Feature and nformation about the game state-------------------------------
        arena = game_state['field']
        explosion = game_state['explosion_map']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        #-----------
        arena_0 = arena[(x,y)]
        arena_1 = arena[(x,y-1)]#up
        arena_2 = arena[(x,y+1)]#down
        arena_3 = arena[(x-1,y)]#left
        arena_4 = arena[(x+1,y)]#right
        #-----------
        explosion_0 = explosion[(x,y)]
        explosion_1 = explosion[(x,y-1)]#up
        explosion_2 = explosion[(x,y+1)]#down
        explosion_3 = explosion[(x-1,y)]#left
        explosion_4 = explosion[(x+1,y)]#right
        #-----------
        coins = game_state['coins']
        coins = [coins[i] for i in range(len(coins)) if coins[i] not in bomb_xys]
        nearestCoins = get_relative_position_to_nearest((x,y), coins, 2)
        [x_nearest_Coin, y_nearest_Coin] = nearestCoins
        #-----------
        crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
        crates = [crates[i] for i in range(len(crates)) if crates[i] not in bomb_xys]
        nearestCrates = get_relative_position_to_nearest((x,y), crates, 2)
        [x_nearest_Crates, y_nearest_Crates] = nearestCrates
        #-----------
        dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        dead_ends = [dead_ends[i] for i in range(len(dead_ends)) if dead_ends[i] not in bomb_xys]
        nearestDead_ends = get_relative_position_to_nearest((x,y), dead_ends, 2)
        [x_nearest_Dead_ends, y_nearest_Dead_ends] = nearestDead_ends
        #-----------
        
        #-----------------------------------------------------------------------------------------------
    # For example, you could construct several channels of equal shape, ...
    #16 features
    channels = [arena_0,arena_1,arena_2,arena_3,arena_4,explosion_0,explosion_1,explosion_2,explosion_3,explosion_4,x_nearest_Coin,y_nearest_Coin,x_nearest_Crates, y_nearest_Crates,x_nearest_Dead_ends, y_nearest_Dead_ends]
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
