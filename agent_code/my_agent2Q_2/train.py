import os
import pickle
import random
from collections import namedtuple, deque
from typing import List
import events as e
from .callbacks import state_to_features
import math
import numpy as np

#1 --------------------------------------------------------------------------------
import pandas as pd
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
#1 --------------------------------------------------------------------------------

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
#2 Define Q learning table ----------------------------------------------------
    self.n_actions = len(ACTIONS);
    self.learning_rate = 0.1
    self.Gamma = 0.9
    if os.path.isfile('q_table.pkl'):
        self.q_table = pd.read_pickle('q_table.pkl')
    else:
        self.q_table = pd.DataFrame(columns=ACTIONS,dtype=np.float64)
#2 ---------------------------------------------------------------------------------


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

#3 new game state, Q table append----------------------------------------------
    observation_new = state_to_features(new_game_state)
    #append new state to q table-------------------------------------------------
    if observation_new not in self.q_table.index:
        self.q_table = self.q_table.append(pd.Series([0]*self.n_actions,index=self.q_table.columns,name=observation_new))
    # print("**************")
    # print(f'Current Nearest coin position:  {nearestCoins}')
       
    
    #old game state--------------------------------------------------------------
    if old_game_state is not None:  # in the first step, the old_game_state is none
        observation_old = state_to_features(old_game_state)
        ########################add reward when next step distance become short to the nearest coins
        # if nearestCoins_new == nearestCoins_old:
            # best_dist_old = np.sum(np.abs(np.subtract(nearestCoins_old, (x_old,y_old))))
            # best_dist_new = np.sum(np.abs(np.subtract(nearestCoins_old, (x_new,y_new))))
            # if best_dist_new < best_dist_old:
                # reward = reward + 1
        ############################################################################################3
        # print("**************")
        # print(f'old agent position: {(x_old,y_old)}')
        #------------------------------------------------------------------------
        #Reward======================================================================
        reward = reward_from_events(self, events)
        #============================================================================     
        #Qtable update------------------------------------------------------------
        if observation_new is not None:#there still have coins
            q_predict = self.q_table.loc[[observation_old],[self_action]].values
            q_target = reward + self.Gamma * self.q_table.loc[[observation_new]].max(1).values
            #print(q_target)
        else:
            q_target = reward  # next state is terminal
        self.q_table.loc[[observation_old],[self_action]] += self.learning_rate * (q_target - q_predict)  # update
    
    #----------------------------------------------------------------------------- 
    #reward = self.reward_from_events(events)
#3 -----------------------------------------------------------------------------
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    #-----------------------------------    
    self.q_table.to_pickle('q_table.pkl')
    #print(self.q_table)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -10,
        e.INVALID_ACTION: -20,
        e.BOMB_DROPPED: -2,
        e.CRATE_DESTROYED: 50,
        #e.COIN_FOUND: 20,
        e.COIN_COLLECTED: 100,
        #e.KILLED_OPPONENT: 300,
        e.KILLED_SELF: -150,
        #e.GOT_KILLED: -150,
        #e.SURVIVED_ROUND: 20,
        e.MOVED_LEFT: -5,
        e.MOVED_RIGHT: -5,
        e.MOVED_UP: -5,
        e.MOVED_DOWN: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum