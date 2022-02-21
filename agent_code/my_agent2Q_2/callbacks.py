import os
import pickle
import random
import numpy as np
import pandas as pd
from random import shuffle
from .RuleBased import act as rule_act
from .RuleBased import setup as rule_setup
from .RuleBased import look_for_targets
import math


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
            self.model = pickle.load(file)
    #-------------------------------------------------------
    if os.path.isfile('q_table.pkl'):
        self.q_table = pd.read_pickle('q_table.pkl')
        #print(self.q_table)
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
    observation = state_to_features(game_state)
    
    #==============================Print=======================================
    # print("***********************************************")
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
            Action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, 0.1, 0.1])
        #print(Action)
        return Action
    #################################################################################   
    else:
        ##otherwise, choose action based on Q table, if there have more action with same Q value, random choose in those action
        if observation not in self.q_table.index:
            Action = rule_act(self, game_state)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        else:
            stateActions = self.q_table.loc[[observation]]
            MaxQval_one = stateActions.max(1).tolist()*len(ACTIONS)# in this state, the MaxValue action in Q table
            MaxQval_multi = np.where(stateActions == MaxQval_one)[1]# in this state, the action with same MaxValue in Q table
            Actions_all = np.array(ACTIONS)[MaxQval_multi.astype(int)]
            Action = np.random.choice(Actions_all)
        #todo Exploration vs exploitation-----------------------------------
        random_prob = .1
        if self.train and random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.20, .20, .20, .20, 0.1, 0.1])
        self.logger.debug("Querying model for action.")
        return Action

#-------------------------------------------------------------------------------------------------------------- 
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
    else:
        ##Gather Feature and nformation about the game state-------------------------------
        arena = game_state['field']
        #explosion = game_state['explosion_map']
        #coordinate of our agent
        _, score, bombs_left, (x_a, y_a) = game_state['self']
        #print(x_a,y_a)
        #coordinate of coins
        coins = game_state['coins']
        #coordinate of bombs and time of explosion left
        bombs = game_state['bombs']
        #coordinate of bombs
        bomb_xys = [xy for (xy, t) in bombs]
        #coordinate of crates
        crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
        #coordinate of dead_ends
        dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                     and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        
        #coordinate of slide window--------------------------------------------
        (x_a0,y_a0) = (x_a-1, y_a-1)
        (x_a1,y_a1) = (x_a, y_a-1)
        (x_a2,y_a2) = (x_a+1, y_a-1)
        (x_a3,y_a3) = (x_a-1, y_a)
        (x_a4,y_a4) = (x_a, y_a)
        (x_a5,y_a5) = (x_a+1, y_a)
        (x_a6,y_a6) = (x_a-1, y_a+1)
        (x_a7,y_a7) = (x_a, y_a+1)
        (x_a8,y_a8) = (x_a+1, y_a+1)
        FeatCord = [(x_a0,y_a0), (x_a1,y_a1), (x_a2,y_a2), (x_a3,y_a3), (x_a4,y_a4), (x_a5,y_a5), (x_a6,y_a6), (x_a7,y_a7), (x_a8,y_a8)]
        #-state corresponding to the coordinate of slide windows--------------------
        
        Feat = [100]*11
        for i in range(0, 9):
            Feat[i] = arena[FeatCord[i]]
            if FeatCord[i] in coins:
                Feat[i] = 2
            elif FeatCord[i] in bomb_xys:
                Feat[i] = 3
        
        #-----------------------------------------------------------------------------------------------------
        #if there are no coins and crate in the slide window, find the nearst coins or crates as target--------
        if (1 or 2) not in Feat:
            #All targets
            targetNext = coins + crates + dead_ends
            if any(targetNext): 
                # remove the targets with bombs
                targetNext = [targetNext[i] for i in range(len(targetNext)) if targetNext[i] not in bomb_xys]
                #find the nearest target
                nearestTarget = targetNext[np.argmin(np.sum(np.abs(np.subtract(targetNext, (x_a, y_a))), axis=1))]
                [x_nearest_target, y_nearest_target] = nearestTarget
                Feat[9] = x_nearest_target - x_a
                Feat[10] = y_nearest_target - y_a
        #-----------------------------------------------------------------------------------------------------
    #11 features
    channels = tuple(Feat)
    #print(channels)
    #print(type(channels))
    
    #channels.append(...)
    #stacked_channels = np.stack(channels)
    return channels
