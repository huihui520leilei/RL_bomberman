import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import random
from collections import namedtuple, deque
from typing import List
import events as e
from .callbacks import state_to_features
from tensorflow import keras
#1 --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
#1 ================================================================================

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
    
#2-Define Parameter ----------------------------------------------------
    self.n_actions = len(ACTIONS)
    self.n_features = 16
    self.learning_rate = 0.1
    self.Gamma = 0.9
    self.replace_target_iter = 200
    self.memory_size = 800000#2000 episodes
    self.batch_size = 42
    #total learning step
    self.learn_step_counter = 0
#---reload and initialize the memory using save (s, a, r, s_), here = 16*2+2 = 34
    if os.path.isfile("memory.pt"):
        self.logger.info("Loading memory from saved memory.")
        with open("memory.pt", "rb") as file:
            self.memory = pickle.load(file)
            #print(self.memory)
    else: 
        self.memory = np.zeros((self.memory_size, self.n_features*2+2))
    #reload and initialize memory_counter-------------------
    if os.path.isfile("memory_counter.pt"):
        self.logger.info("Loading memory counter from saved.")
        with open("memory_counter.pt", "rb") as file:
            self.memory_counter = pickle.load(file)
            #print(self.memory_counter)
    else: 
        self.memory_counter = 0
    #reload and initialize cost_his-------------------
    if os.path.isfile("cost_his.pt"):
        self.logger.info("Loading cost_history from saved.")
        with open("cost_his.pt", "rb") as file:
            self.cost_his = pickle.load(file)
    else: 
        self.cost_his = []
    #-----initialize DQN model-------------------------------  
    build_net(self)
    self.sess = tf.Session() 
    self.sess.run(tf.global_variables_initializer())  
#---use update the target and eval network-------
    self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
    self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')    
    with tf.variable_scope('hard_replacement'):
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
#2=============================================================================
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
#3-Transitions store---------------------------------------------------------------------------
    #Reward----------------------------------
    reward = reward_from_events(self, events)
    #action to int-----------------------------------
    if self_action is not None:
        ac = ACTIONS.index(self_action)
    else:
        print('self_Action is none')
        ac = 0
    #store_transition(s, a, r, s_)------------------- 
    s = state_to_features(old_game_state)
    a = ac
    r = reward
    s_= state_to_features(new_game_state)
    #------------------------------------------------
    transition = np.hstack((s, a, r, s_))
    #replace the old memory with new memory
    index = self.memory_counter % self.memory_size
    self.memory[index, :] = transition
    self.memory_counter += 1
#3=============================================================================================

#4-Learn----------------------------------------------------------------------------------------
#---check to replace target parameters------------------
    if self.learn_step_counter % self.replace_target_iter == 0:
        self.CheckReplace = self.sess.run(self.target_replace_op)
        # print(self.CheckReplace)
        print('\nTarget_params_replaced\n')
        print("***********************************")

#---sample batch memory from all memory-----------------   
    if self.memory_counter > self.memory_size:
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    else:
        sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
    batch_memory = self.memory[sample_index, :]
#---compute cost--------------------------------------- 
    _, cost = self.sess.run(
        [self._train_op, self.loss],
        feed_dict={
            self.s: batch_memory[:, :self.n_features],
            self.a: batch_memory[:, self.n_features],
            self.r: batch_memory[:, self.n_features + 1],
            self.s_: batch_memory[:, -self.n_features:],
        })
#-----record cost -------------------------------------        
    self.cost_his.append(cost)
    self.learn_step_counter += 1 
#---check q_next and q_eval--------------------------
    # self.Q_next, self.Q_eval = self.sess.run(
        # [self.q_next, self.q_eval],
        # feed_dict={
            # self.s: batch_memory[:, :self.n_features],
            # self.s_: batch_memory[:, -self.n_features:],
        # })    
#-Use this to test DQN_model---------------------------------------------
    # observation = state_to_features(old_game_state)
    # observation = observation[np.newaxis, :]
    # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
    # action = np.argmax(actions_value)        
    # Action = ACTIONS[action]
    # print("**********************")
    # print(Action)
    # print("=======2=======")
    # print(self.Q_next.shape)
    # print("=======2=======")
#4==============================================================================================
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
#----------------------------------------------------------------------------------------------
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

#5-Store the model------------------------------------------------------------------------------
    # save Eval Network model parameters-------------------------
    self.evalN = self.sess.run(self.e_params)
    # weight_L1 = np.array(self.evalN[0],dtype=object)
    # bias_L1 = np.array(self.evalN[1],dtype=object)
    # weight_L2 = np.array(self.evalN[2],dtype=object)
    # bias_L2 = np.array(self.evalN[3],dtype=object)    
    # print("=======1=======")
    # print(weight_L1.shape)
    # print(bias_L1)
    # print(weight_L2.shape)
    # print(bias_L2.shape)
    # print("=======1=======")
    # save momery cost_history-------------------------------------    
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.evalN, file)
    with open("memory.pt", "wb") as file:
        pickle.dump(self.memory, file)
    with open("memory_counter.pt", "wb") as file:
        pickle.dump(self.memory_counter, file)
    with open("cost_his.pt", "wb") as file:
        pickle.dump(self.cost_his, file)
#5=======================================================================

#6-plot cos---------------------------------------------------------------- 
    # plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    # plt.ylabel('Cost')
    # plt.xlabel('training steps')
    # plt.show()
#-----spare code for save-----------------------------------------------
    # saver = tf.train.Saver(self.e_params)
    # saver.save(self.sess, './EvalModel/model', write_meta_graph=False)
    # #-check------------------------------------------------
    # self.evalN = self.sess.run(self.e_params)
    # print("=======2=======")
    # print(self.evalN)
    # print("=======2=======") 
#6=============================================================================================================================== 
def build_net(self):
    #all inputs------------------------------------
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
    self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
    self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
    self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
    if os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading cost_history from saved.")
        with open("my-saved-model.pt", "rb") as file:
            self.DQNpara = pickle.load(file)
            weight_L1 = np.array(self.DQNpara[0],dtype=object)
            bias_L1 = np.array(self.DQNpara[1],dtype=object)
            weight_L2 = np.array(self.DQNpara[2],dtype=object)
            bias_L2 = np.array(self.DQNpara[3],dtype=object)    
            # print("=======2=======")
            # print(weight_L1.shape)
            # print(bias_L1)
            # print(weight_L2.shape)
            # print(bias_L2.shape)
            # print("=======2=======")
            w1_initializer, b1_initializer = tf.constant_initializer(weight_L1), tf.constant_initializer(bias_L1)
            w2_initializer, b2_initializer = tf.constant_initializer(weight_L2), tf.constant_initializer(bias_L2)
    else:
        w1_initializer, b1_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        w2_initializer, b2_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)                
        print("*****random_normal_initializer weight and bias*****")

    #build evaluate_net------------------------------
    with tf.variable_scope('eval_net'):
        e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w1_initializer,
                             bias_initializer=b1_initializer, name='e1')
        self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w2_initializer,
                                      bias_initializer=b2_initializer, name='q')
    
    #build target_net--------------------------------
    with tf.variable_scope('target_net'):
        t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w1_initializer,
                             bias_initializer=b1_initializer, name='t1')
        self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w2_initializer,
                                      bias_initializer=b2_initializer, name='t2')

    with tf.variable_scope('q_target'):
        q_target = self.r + self.Gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
        self.q_target = tf.stop_gradient(q_target)
    with tf.name_scope('q_eval'):
        a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
        self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
    with tf.variable_scope('loss'):
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
    with tf.variable_scope('train'):
        self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
#==================================================================================================

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -2,
        e.INVALID_ACTION: -20,
        e.BOMB_DROPPED: -5,
        e.CRATE_DESTROYED: 50,
        e.COIN_FOUND: 20,
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 300,
        e.KILLED_SELF: -500,
        e.GOT_KILLED: -500,
        e.SURVIVED_ROUND: 100
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
#===========================================================================================