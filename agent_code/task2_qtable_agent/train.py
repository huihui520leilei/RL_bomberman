import pickle
import random
import os.path
from collections import namedtuple, deque
from typing import List
import numpy as np
import pandas as pd
import events as e
from .callbacks import state_to_features, get_goal_information_from_game_state
from .auxiliary import TASK2_ACTIONS, TASK2_QTABLE_SIZE
from .auxiliary import learning_rate, discount
from .auxiliary import init_rand_min, init_rand_max

from .auxiliary import features_to_string

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if os.path.isfile('q_table.npy'):
        self.q_table = np.load('q_table.npy')
    else:
        self.q_table = np.random.uniform(low=init_rand_min, high=init_rand_max, size=TASK2_QTABLE_SIZE)
    
    if os.path.isfile('train_log.csv'):
        self.log_df = pd.read_csv('train_log.csv')
    else:
        self.log_df = pd.DataFrame(columns=['round', 'step', 'coins_left'])
    



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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    
    (up_old, down_old, right_old, left_old, x_next_goal_old, y_next_goal_old, goal_old) = state_to_features(old_game_state)
    (up,    down,      right,     left,     x_next_goal,     y_next_goal,     goal    ) = state_to_features(new_game_state)
    
    # compute distance to goal
    goal_dist_old = np.sqrt(x_next_goal_old**2+y_next_goal_old**2)
    goal_dist_new = np.sqrt(x_next_goal**2    +y_next_goal**2    )

    # get mode
    (mode_old, _, _, _) = get_goal_information_from_game_state(old_game_state)
    (mode_new, _, _, _) = get_goal_information_from_game_state(new_game_state)



    if self_action is not None:
        action_index = TASK2_ACTIONS.index(self_action)
    else:
        action_index = 1




    reward = 0.0
    if (e.INVALID_ACTION in events):
        reward = reward - 3.0

    #if goal_old == 0:
    #    reward += goal_dist_new/2.0 # reward for having great distance to bombs and explosions

    if goal_dist_new < goal_dist_old:
        reward += mode_old
    if goal_dist_new > goal_dist_old:
        reward -= mode_old
    
    if (e.BOMB_DROPPED in events):
        if (goal_old==1) and (goal_dist_old==1):
            reward += 10 # that is a good bomb
        else:
            reward -= 10 # we do not want bombs otherwise
    
    if (e.GOT_KILLED in events):
        reward += -100

    if (goal_old == 0) and (e.CRATE_DESTROYED in events):
        reward += 10
    
    if (goal_old==1) and (e.COIN_COLLECTED in events):
        reward = reward + 5.0

    if (mode_old != mode_new):
        reward += 2

    reward = reward - 1.0 # costs for moving
        
    current_q = self.q_table[(up_old, down_old, right_old, left_old, x_next_goal_old, y_next_goal_old, goal_old, action_index)]
    v = np.max(self.q_table[(up, down, right, left, x_next_goal, y_next_goal, goal)]) # q-learning
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * v)
    self.q_table[(up_old, down_old, right_old, left_old, x_next_goal_old, y_next_goal_old, goal_old, action_index)] = new_q

    
    #print(f'min q-value: {np.min(self.q_table)}   maximum q-value: {np.max(self.q_table)}')
    #self.q_table = (self.q_table - np.min(self.q_table))/np.ptp(self.q_table) # normalise to [0,1]
    #self.q_table = 2.*(self.q_table - np.min(self.q_table))/np.ptp(self.q_table)-1 # normalise to [-1,1]
    

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
        step = last_game_state['step']
        print(f'coin(s) left: {len(coins_left)} step: {step}')
        
        # if the last action / state was not successful, lower q-value
        (up_last, down_last, right_last, left_last, x_next_coin_last, y_next_coin_last, goal_last) = state_to_features(last_game_state)
        action_index = TASK2_ACTIONS.index(last_action)
        self.q_table[(up_last, down_last, right_last, left_last, x_next_coin_last, y_next_coin_last, goal_last, action_index)] = self.q_table[(up_last, down_last, right_last, left_last, x_next_coin_last, y_next_coin_last, goal_last, action_index)] - 0.3 # 0.3
    np.save('q_table.npy', self.q_table)

    round = last_game_state['round']
    step = last_game_state['step']
    coins_left = len(coins_left)
    new_entry = pd.DataFrame({'round': [round], 'step': [step], 'coins_left': [coins_left]})
    self.log_df = pd.concat([self.log_df, new_entry])

    self.log_df.to_csv('train_log.csv', index=False)








def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum