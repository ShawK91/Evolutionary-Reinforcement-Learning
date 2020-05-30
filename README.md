
The master branch introduces many additions over the NeurIPS paper published in 2018 - improving significantly on runtime and algorithmic performance. The primary changes are detailed below:
1. Parallleized rollouts
2. Soft-Actor Critic (SAC) https://arxiv.org/abs/1801.01290 replacing the DDPG used originally
3. Support for Discrete Environments using a form of DDQN + Maximum Entropy Reinforcement Learning

Please switch to neurips_paper_2018 branch if you wish to reproduce the original results from the paper https://papers.nips.cc/paper/7395-evolution-guided-policy-gradient-in-reinforcement-learning.pdf


## Dependencies Tested on ##
Python 3.6.9 \
Pytorch 1.2 \
Numpy 1.18.1 \
Gym 0.15.6 \
Mujoco-py v1.50.1.59

## To Run ##
python main.py --env $ENV_NAME$ 

## Environment name examples to get you started ##

#### Continous ###
'Humanoid-v2' \
'Hopper-v2' \
'HalfCheetah-v2' \
'Swimmer-v2' \
'Ant-v2' \
'Walker2d-v2' \
'Reacher-v2' \

#### Discrete ####
'CartPole-v1' \
'Pong-ram-v0' \
'Qbert-ram-v0' \
'MountainCar-v0' 

## To use your own custom environment ##

Write a gym-compatible wrapper around your environment and register it with the gym runtime  