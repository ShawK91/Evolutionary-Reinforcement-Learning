import numpy as np, os, time, random
from envs_repo.constructor import EnvConstructor
from models.constructor import ModelConstructor
from core.params import Parameters
import argparse, torch
from algos.erl_trainer import ERL_Trainer

parser = argparse.ArgumentParser()

#######################  COMMANDLINE - ARGUMENTS ######################
parser.add_argument('--env', type=str, help='Env Name',  default='Pendulum-v0')
parser.add_argument('--seed', type=int, help='Seed', default=991)
parser.add_argument('--savetag', type=str, help='#Tag to append to savefile',  default='')
parser.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)
parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=2)
parser.add_argument('--buffer', type=float, help='Buffer size in million',  default=1.0)
parser.add_argument('--frameskip', type=int, help='Frameskip',  default=1)

parser.add_argument('--hidden_size', type=int, help='#Hidden Layer size',  default=256)
parser.add_argument('--critic_lr', type=float, help='Critic learning rate?', default=3e-4)
parser.add_argument('--actor_lr', type=float, help='Actor learning rate?', default=1e-4)
parser.add_argument('--tau', type=float, help='Tau', default=1e-3)
parser.add_argument('--gamma', type=float, help='Discount Rate', default=0.99)
parser.add_argument('--alpha', type=float, help='Alpha for Entropy term ',  default=0.1)
parser.add_argument('--batchsize', type=int, help='Seed',  default=512)
parser.add_argument('--reward_scale', type=float, help='Reward Scaling Multiplier',  default=1.0)
parser.add_argument('--learning_start', type=int, help='Frames to wait before learning starts',  default=5000)

#ALGO SPECIFIC ARGS
parser.add_argument('--popsize', type=int, help='#Policies in the population',  default=10)
parser.add_argument('--rollsize', type=int, help='#Policies in rollout size',  default=5)
parser.add_argument('--gradperstep', type=float, help='#Gradient step per env step',  default=1.0)
parser.add_argument('--num_test', type=int, help='#Test envs to average on',  default=5)

#Figure out GPU to use [Default is 0]
os.environ['CUDA_VISIBLE_DEVICES']=str(vars(parser.parse_args())['gpu_id'])

#######################  Construct ARGS Class to hold all parameters ######################
args = Parameters(parser)

#Set seeds
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

################################## Find and Set MDP (environment constructor) ########################
env_constructor = EnvConstructor(args.env_name, args.frameskip)


#######################  Actor, Critic and ValueFunction Model Constructor ######################
model_constructor = ModelConstructor(env_constructor.state_dim, env_constructor.action_dim, args.hidden_size)


ai = ERL_Trainer(args, model_constructor, env_constructor)
ai.train(args.total_steps)


