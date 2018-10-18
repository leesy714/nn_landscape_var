import argparse
import os

import numpy as np
import torch

from .model import Policy
from .envs import VecPyTorch, make_vec_envs
from .utils import get_render_func, get_vec_normalize



def evaluate(env_name, filename, seed = 1,add_timestep=False,det = True):
    env = make_vec_envs(env_name, seed + 1000, 1,
                                None, None, add_timestep, device='cpu',
                                allow_early_resets=False)
    state_dict, ob_rms = torch.load(filename)
    actor_critic = Policy(env.observation_space.shape, env.action_space,
        base_kwargs = {'recurrent':False})
    #print(state_dict)
    actor_critic.load_state_dict(state_dict)
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    obs = env.reset()
    total_r = 0.0
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=det)
        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        total_r += reward
        if done:
            break
    return total_r.data.item()

def evaluate_net(net, env, seed=1,max_iter=3000, det=True):
    recurrent_hidden_states = torch.zeros(1, net.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    obs = env.reset()
    total_r = 0.0
    for i in range(max_iter):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = net.act(
                obs, recurrent_hidden_states, masks, deterministic=det)
        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        total_r += reward
        if done:
            break
    return total_r.data.item()


    


if __name__=='__main__':
    env_name = 'BeamRiderNoFrameskip-v4'
    filename = './trained_models/BeamRiderNoFrameskip-v4.pt'
    print(evaluate(env_name, filename))
