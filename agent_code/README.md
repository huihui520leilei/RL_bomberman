# Task 1: Only Coins
## Folder: my_agent1Q (use off-policy Q-table)
Running code:
```console
python main.py play --agents my_agent1Q
```
## Folder: my_agent1Sarsa (use on-policy SARSA)
Running code:
```console
python main.py play --agents my_agent1Sarsa
```
## training file: train_my_agent.py
1. Copy folder "my_agent1Q" under "/bomberman_rl/agent_code" 
2. Copy "train_my_agent.py" under "/bomberman_rl"

# Task 2: Without Opponents
## 1. Deep Q Network (folder: my_agent2DQN)
1) Finish DQN Framework in "callbacks.py" and "train.py"
2) 1 hidden layer with 64 nodes
3) 2 hidden layer with 32 nodes for everyone
Running code:
```console
python main.py play --agents my_agent2DQN --train 1
```
## 2. New state based on Q-table (folder: my_agent2Q_2)
1) works on task 1 does't work on task 2 (the model saved in 'my_agent2Q' is task 1 table trained based on the new state)
    
