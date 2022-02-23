# ML_project

Repository for the final project of the lecture "Fundamentals of Machine Learning" of Team "Looks Cute".
******************************************************************************
### Agent:
![Alt text](agentFig/avatar1.png?raw=true)  
### Bomb:
![Alt text](agentFig/bomb.png?raw=true)


## Materials:
###  1. Deep Q Learning
(1) Tutorial Vedio：[Deep Reinforcement Learning](https://www.youtube.com/watch?v=z95ZYgPgXOY&list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)

(2) [Flappy Bird Using Deep Reinforcement Learning](https://github.com/floodsung/DRL-FlappyBird)

(3) [Using Deep Q-Network to Learn How To Play Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird)

(4) [Maze Using Reinforcement Learning(Q_learning, Sarsa, DQN)](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents)


******************************************************************************
******************************************************************************
# Version(Linux)
******************************************************************************
******************************************************************************
## Create conda environment
```console
conda create --name ml_project python=3.8
conda activate ml_project
pip install -r ML_project/requirements.txt
```

## Clone repositories
```console
git clone git@github.com:holger-ziti/ML_project.git
git clone git@github.com:ukoethe/bomberman_rl.git
```

## Copy agent-folders
```console
cp -r ./ML_project/agent_code/number1a_agent ./bomberman_rl/agent_code
```
## Play
```console
conda activate ml_project
cd bomberman_rl
python main.py play
conda deactivate
```

# Task 1 (collect all coins)

## Symbolic link (better than copy for development):
```console
rm -r ~/git/bomberman_rl/agent_code/number1a_agent/
ln -s ~/git/ML_project/agent_code/number1a_agent/ ~/git/bomberman_rl/agent_code/
```

## Change the environment
see chapter 4 in [final_project.pdf](/final_project.pdf): 
in settings.py: CRATE_DENSITY = 0 for empty field


let our agent play
```console
python main.py play --agents number1a_agent
```

let our agent train
```console
python main.py play --agents number1a_agent --train 1
```
# Task 2 
in settings.py: CRATE_DENSITY = 0.75 for crates
```console
ln -s ~/git/ML_project/agent_code/number2b_agent/ ~/git/bomberman_rl/agent_code/
python main.py play --agents number2b_agent
```

## How to train agent 2b
- in folder [/agent_code/number2b_agent/](/agent_code/number2b_agent/) delete the following files: [memory.sav](/agent_code/number2b_agent/memory.sav) and [model.sav](/agent_code/number2b_agent/model.sav)
- adapt the rewards and parameters in [parameters.py](/agent_code/number2b_agent/parameters.py) 
- got to folder [/training/](/training/)
- adapt the number of episodes you want to train in [do_training.py](/training/do_training.py)
- run training with command: *python do_training.py*
- if the model is good, copy [parameters.py](/agent_code/number2b_agent/parameters.py), [memory.sav](/agent_code/number2b_agent/memory.sav) and [model.sav](/agent_code/number2b_agent/model.sav) to folder [/agent_code/number2b_agent/saved/](/agent_code/number2b_agent/saved/).

# Other commands
control via keyboard:
```console
python main.py play --agents user_agent
```

******************************************************************************
******************************************************************************
# Version(Windows)
******************************************************************************
******************************************************************************
## Create conda environment
1 Open and Lunch "Anaconda Prompt(Anaconda3)" in Anaconda3(64-bit)
2 If it is the first time to do the project, do:
```console
1) conda create --name ml_project python=3.8
2) conda activate ml_project
3) pip install (the library in requirements.txt)
4) build a new folder named “mlclass_project” in your desktop
```

## Clone repositories
```console
1) git clone git@github.com:holger-ziti/ML_project.git
2) git clone git@github.com:ukoethe/bomberman_rl.git
3) Move them all in the folder “mlclass_project”
```


## Play
1. input command in "CMD.exe Prompt": activate, run rule_based_agent example, deactivate
```console
1) conda activate ml_project
2) cd desktop
    cd mlclass_project
    cd bomberman_rl
3) python main.py play
4) conda deactivate
```
2. Play one defined agent (in bomberman_rl/agent_code) with three rule_based_agent
```console
python main.py play --my-agent random_agent
```
3. Play only one agent in （bomberman_rl/agent_code）
```console
python main.py play --agents peaceful_agent
```
******************************************************************************
## Task 1 (Collect all coins without crates and opponents)(Q-table and Sarsa Method)（finished）
### 1. Create your own agent

Copy "agent_code/tpl_agent" rename as "my_agent"

### 2. Change the environment

1). Change in "settings.py": CRATE_DENSITY = 0 for empty field, 

2). Change in "callbacks.py": ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT'], Random choice: p=[.25, .25, .25, .25]

### 3. Training model based on rule_based_agent 

1). If training run just one times, input in "CMD.exe Prompt":

```console
python main.py play --agents my_agent --train 1
```
2). If the number of training runs depends on the "episodes" in "train_my_agent.py", input in "CMD.exe Prompt":
```console
python train_my_agent.py
```
### 4. Running Q Learning model for action
1). Input in "CMD.exe Prompt":
```console
python main.py play --agents my_agent1Q
```
******************************************************************************
## Task 2 (Collect all coins and bomb crates without opponents)
1). Material:
Deep_Q_Network
[Maze Using Reinforcement Learning(Q_learning, Sarsa, DQN)](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents)

## 1. Neural Network (folder: my_agent2)
## (Training file: ......)
2) Finish DQN Framework in "callbacks.py" and "train.py"

Running code:
```console
python main.py play --agents my_agent2 --train 1
```
## 2. New state based on Q-table (folder: my_agent2Q_2)
1) works on task 1 does't work on task 2

