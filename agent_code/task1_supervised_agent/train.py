import pickle
import random
import os.path
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from .auxiliary import TASK1_ACTION_TABLE_SIZE
from .rule_based_callbacks import act as rb_act
from .coin_collector_angent_callbacks import coin_act

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if os.path.isfile('X.npy'):
        self.X = np.load('X.npy')
    else:
        self.X = np.zeros((0, 6))

    if os.path.isfile('y.npy'):
        self.y = np.load('y.npy')
    else:
        self.y = np.zeros((0, 1))


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
    
    if old_game_state is None:
        return

    #(up_old, down_old, right_old, left_old, x_next_coin_old, y_next_coin_old) = state_to_features(old_game_state)
    #(up, down, right, left, x_next_coin, y_next_coin) = state_to_features(new_game_state)

    features = state_to_features(old_game_state)

    #action = rb_act(self, old_game_state) # string
    action = coin_act(self, old_game_state)
    action_index = ACTIONS.index(action)


    self.X = np.vstack([self.X, features])
    self.y = np.vstack([self.y, action_index])



    #print(self.X.shape)
    #print(self.y.shape)

    #self.action_table[(up, down, right, left, x_next_coin, y_next_coin)] = action_index
    #self.action_table[(up, down, right, left, x_next_coin, y_next_coin, action_index)] = self.action_table[(up, down, right, left, x_next_coin, y_next_coin, action_index)] + 1


    #    action_index = TASK1_ACTIONS.index(self_action)
    #else:
    #    #print("action is none")
    #    action_index = 1

    #reward = 0.0
    #if (e.INVALID_ACTION in events):
    #	reward = reward - 3.0
    #	#print('invalid action')

    #if (e.COIN_COLLECTED in events):
    #    reward = reward + 20.0
    #    #print("coin collected")
    #else:
    #    reward = reward - 1.0 # costs for moving
    #    #print(f'coin collected. mean: {np.mean(self.q_table)}')
    #    #self.q_table[(up_old, down_old, right_old, left_old, x_next_coin_old, y_next_coin_old, action_index)] = 1.0
    #current_q = self.q_table[(up_old, down_old, right_old, left_old, x_next_coin_old, y_next_coin_old, action_index)]
    #v = np.max(self.q_table[(up, down, right, left, x_next_coin, y_next_coin)]) # q-learning
    #v = self.q_table[(up, down, right, left, x_next_coin, y_next_coin, action_index)] # sarsa alg.
    #new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * v)
    #self.q_table[(up_old, down_old, right_old, left_old, x_next_coin_old, y_next_coin_old, action_index)] = new_q
    #print(f"new_q is: {new_q}")
    
    #if (e.COIN_COLLECTED in events):
    #    print(f"current_q: {current_q} - new_q: {new_q}")
    #print(f'min q-value: {np.min(self.q_table)}   maximum q-value: {np.max(self.q_table)}')
    #self.q_table = (self.q_table - np.min(self.q_table))/np.ptp(self.q_table) # Normalised [0,1]
    #self.q_table = 2.*(self.q_table - np.min(self.q_table))/np.ptp(self.q_table)-1 # Normalised [-1,1]
    

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


    self.model.fit(self.X,self.y.ravel())
    print(self.model.score(self.X, self.y, sample_weight=None))
    print(self.X.shape)
    print(self.y.shape)


    with open("model.pt", "wb") as file:
        pickle.dump(self.model, file)

    np.save('X.npy', self.X)
    np.save('y.npy', self.y)




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
