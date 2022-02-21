# Task 1: Only Coins
## folder: my_agent (use off-policy Q-table)
Running code:
```console
python main.py play --agents my_agent
```
## folder: my_agent_sarsa (use on-policy SARSA)
Running code:
```console
python main.py play --agents my_agent_sarsa
```
## training file: train_my_agent.py
1. Copy folder "my_agent" under "/bomberman_rl/agent_code" 
2. Copy "train_my_agent.py" under "/bomberman_rl"

# Task 2: Without Opponents
## 1. Neural Network (folder: my_agent2)
## (Training file: ......)
1) Finish DQN Framework in "callbacks.py" and "train.py"

Running code:
```console
python main.py play --agents my_agent2 --train 1
```
## 2. New state based on Q-table (folder: my_agent2Q_2)
1) works on task 1 does't work on task 2 (the model saved in 'my_agent2Q_2' is task 1 table trained based on the new state)
    
