import gym
import torch
from functools import reduce
from operator import mul
from maml.policy import CategoricalMLPPolicy, NormalMLPPolicy

def get_policy_for_env(env):
    # nonlinearity = getattr(torch, nonlinearity)
    # if isinstance(env.action_space, gym.spaces.Box):
    #     output_size = reduce(mul, env.action_space.shape, 1)
    #     policy = NormalMLPPolicy(input_size, output_size, hidden_sizes=tuple(hidden_sizes), nonlinearity=nonlinearity)
    # else:
    #     output_size = env.action_space.n
    #     policy = CategoricalMLPPolicy(input_size, output_size, hiddensizes=tuple(hidden_sizes), nonlinearity=nonlinearity)
    # return policy
    obs_shape = env.observation_space[0].shape
    act_shape = env.action_space[0].nvec
    policy = CategoricalMLPPolicy(obs_shape, act_shape, env.n_clusters)
    return policy