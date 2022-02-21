#====run "python train_my_agent.py" for train========================================


import os
from time import time

#mins = 4 # 4 episodes per min
episodes = 100
counter = 0
t0 = time()

stop = False
for n in range(episodes):
	try:
		os.chdir(r"C:\Users\hui\Desktop\mlclass_project\bomberman_rl")
		#os.system("python main.py play --agents my_agent --train 1")
		os.system("python main.py play --agents my_agent --train 1 --no-gui")
		time_delta = time() - t0
		counter = counter+1
		print(f"number {counter} of {episodes} runtime: {time_delta}")
	except KeyboardInterrupt:
		print("Press Ctrl-C to terminate while statement")
		stop = True	
	if stop:
		break
