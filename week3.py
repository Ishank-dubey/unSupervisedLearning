#Reinforcement learning
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utilsOld
import utils
#from pyvirtualdisplay import Display

#The environment is considered solved if you get 200 points.

# Set up a virtual display to render the Lunar Lander environment.
#Display(visible=0, size=(840, 480)).start();

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

print('OOOK')
#The agent has four discrete actions available:
#Do nothing= 0
#Fire right engine = 1
#Fire main engine = 2
#Fire left engine = 3

#Observation space
#Its linear velocities  (𝑥˙,𝑦˙)

#Its angle  𝜃
#Its angular velocity  𝜃˙
#Two booleans,  𝑙 and  𝑟
#that represent whether each leg is in contact with the ground or not.

#Rewards
#After every step, a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

#For each step, the reward:

#is increased/decreased the closer/further the lander is to the landing pad.
#is increased/decreased the slower/faster the lander is moving.
#is decreased the more the lander is tilted (angle not horizontal).
#is increased by 10 points for each leg that is in contact with the ground.
#is decreased by 0.03 points each frame a side engine is firing.
#is decreased by 0.3 points each frame the main engine is firing.
#The episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively.
env = gym.make('LunarLander-v2')

#In the standard “agent-environment loop” formalism, an agent interacts with the environment in discrete time steps  𝑡=0,1,2,...
#At each time step  𝑡
#the agent uses a policy  𝜋
#to select an action  𝐴𝑡
#based on its observation of the environment's state  𝑆𝑡
#The agent receives a numerical reward  𝑅𝑡
#and on the next time step, moves to a new state  𝑆𝑡+1

