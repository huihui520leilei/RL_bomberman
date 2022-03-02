import os
import pickle
import random
import numpy as np

from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier

from .coin_collector_angent_callbacks import coin_act

from .auxiliary import get_relative_position_to_nearest_coin, get_surrounding
from .auxiliary import relative_position_to_target, look_for_targets_coin_collector
from .auxiliary import max_coin_distance
from .rule_based_callbacks import setup as rb_setup
from .rule_based_callbacks import act as rb_act



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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

    if os.path.isfile("model.pt"):
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)
            print("model loaded from file")
        self.trained = True
    else:
        self.model =  LogisticRegression(max_iter=1000) #MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,100,100))
        self.trained = False

    rb_setup(self)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    features = state_to_features(game_state)



    if self.train:
        action = coin_act(self, game_state)
        #action = rb_act(self, game_state) # string

    elif self.trained:
        #action_index = self.model.predict(features)
        #action = ACTIONS[action]
        #print(action)

        probabilities = self.model.predict_proba(features.reshape(1,-1))
        print(probabilities.shape)

        action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'], p=probabilities[0])
        print(f"chosen action: {action}")
    else: 
        action = np.random.choice(ACTIONS, p=[.20, .20, .20, .20, 0.1, 0.1])


    
    return action





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
        return (0, 0, 0, 0, 0, 0) # in the first step, the old_game_state is none

    else:
        # Gather information about the game state (next 11 lines of code taken from rule-based agent)
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        #if len(coins)==1: #is never empty in task 1
        #    print(f"one coin left")
        #else:
        #    print(coins)
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        [x_next_coin, y_next_coin] = get_relative_position_to_nearest_coin((x, y), coins, max_coin_distance)
        [up, down, right, left] = get_surrounding((x, y), arena) # == 1 if occupied
        features = (up, down, right, left, x_next_coin, y_next_coin) #features are indexes for q-table
        #return np.array(features)


        free_space = arena == 0
        target = look_for_targets_coin_collector(free_space, (x, y), coins)
        [x_next_coin, y_next_coin] = relative_position_to_target(target, (x,y))
        features = (up, down, right, left, x_next_coin, y_next_coin) #features are indexes for q-table
        return np.array(features)
