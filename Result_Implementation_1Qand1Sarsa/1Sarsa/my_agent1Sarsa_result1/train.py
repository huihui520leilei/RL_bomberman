import os
import pickle
import random
from collections import namedtuple, deque
from typing import List
import events as e
from .callbacks import state_to_features

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
    if os.path.isfile('train_log.csv'):
        self.log_df = pd.read_csv('train_log.csv')
    else:
        self.log_df = pd.DataFrame(columns=['round', 'step', 'coins_Left'])
    
#2 Define Q learning table ----------------------------------------------------

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

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    coins_left = last_game_state['coins']
    if not (len(coins_left) <= 1 and (e.COIN_COLLECTED in events)):
        print(f'coin(s) left: {len(coins_left)}')

    roundN = last_game_state['round']
    step = last_game_state['step']
    coins_left = len(coins_left)
    new_entry = pd.DataFrame({'round': [roundN], 'step': [step], 'coins_left': [coins_left]})
    self.log_df = pd.concat([self.log_df, new_entry])
    self.log_df.to_csv('train_log.csv', index=False)