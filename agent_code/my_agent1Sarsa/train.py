import os
import pickle
import random
from collections import namedtuple, deque
from typing import List
import events as e
from .callbacks import state_to_features
from .RuleBased import act as rule_act

#1 --------------------------------------------------------------------------------
import pandas as pd
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']#, 'WAIT', 'BOMB']
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
    if os.path.isfile('q_table.pickle'):
        with open('q_table.pickle', 'rb') as f:
            self.q_table = pickle.load(f)
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
    _, score, bombs_left, (x_new, y_new) = new_game_state['self']
    coins_new = new_game_state['coins']
    # print("**************")
    # print(f'Current agent position: {(x_new,y_new)}')
    #Find coordinate of closest targets
    if any(coins_new):
        targets_new = coins_new
        nearestCoins_new = targets_new[np.argmin(np.sum(np.abs(np.subtract(targets_new, (x_new,y_new))), axis=1))]
        [x_nearest_coin_new, y_nearest_coin_new] = nearestCoins_new
        #append new state to q table-------------------------------------------------
        if (x_new, y_new, x_nearest_coin_new, y_nearest_coin_new) not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0]*self.n_actions,index=self.q_table.columns,name=(x_new, y_new, x_nearest_coin_new, y_nearest_coin_new)))
        # print("**************")
        # print(f'Current Nearest coin position:  {nearestCoins}')
        
        #old game state--------------------------------------------------------------
        if old_game_state is not None:  # in the first step, the old_game_state is none
            _, score, bombs_left, (x_old, y_old) = old_game_state['self']
            coins_old = old_game_state['coins']
            targets_old = coins_old
            nearestCoins_old = targets_old[np.argmin(np.sum(np.abs(np.subtract(targets_old, (x_old,y_old))), axis=1))]
            [x_nearest_coin_old, y_nearest_coin_old] = nearestCoins_old
            #new game state Corresponding Action=========================================
            if (x_new, y_new, x_nearest_coin_new, y_nearest_coin_new) not in self.q_table.index:
                Action_new = rule_act(self, new_game_state)
            else:
                stateActions_new = self.q_table.loc[[(x_new, y_new, x_nearest_coin_new, y_nearest_coin_new)]]
                MaxQval_one_new = stateActions_new.max(1).tolist()*len(ACTIONS)# in this state, the MaxValue action in Q table
                MaxQval_multi_new = np.where(stateActions_new == MaxQval_one_new)[1]# in this state, the action with same MaxValue in Q table
                Actions_all_new = np.array(ACTIONS)[MaxQval_multi_new.astype(int)]
                Action_new = np.random.choice(Actions_all_new)
            #Reward======================================================================
            reward = reward_from_events(self, events)
            #============================================================================     
            #Qtable update------------------------------------------------------------
            q_predict = self.q_table.loc[[(x_old,y_old,x_nearest_coin_old, y_nearest_coin_old)],[self_action]].values
            if targets_new is not None:#there still have coins
                q_target = reward + self.Gamma * self.q_table.loc[[(x_new, y_new, x_nearest_coin_new, y_nearest_coin_new)],[Action_new]].values
                #print(self.q_table.loc[[(x_new, y_new, x_nearest_coin_new, y_nearest_coin_new)],[Action_new]].values)
            else:
                q_target = reward  # next state is terminal
            self.q_table.loc[[(x_old,y_old,x_nearest_coin_old, y_nearest_coin_old)],[self_action]] += self.learning_rate * (q_target - q_predict)  # update
        
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
    with open("q_table.pickle", "wb") as file:
        pickle.dump(self.q_table, file) 
    #print(self.q_table)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -2,
        e.INVALID_ACTION: -3,
        #e.BOMB_DROPPED: -20,
        #e.CRATE_DESTROYED: 100,
        #e.COIN_FOUND: 100,
        e.COIN_COLLECTED: 100,
        #e.KILLED_OPPONENT: 300,
        #e.KILLED_SELF: -300,
        #e.GOT_KILLED: -200,
        #e.SURVIVED_ROUND: 50MOVED_LEFT = 'MOVED_LEFT'
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum