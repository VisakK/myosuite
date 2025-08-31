from stable_baselines3 import PPO, SAC
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
from myosuite.utils import gym
from stable_baselines3.common.policies import ActorCriticPolicy

policy = SAC.load("./policies/stand_policy_p1_SAC")

step = 0 
env = gym.make('myoChallengeSoccerP1-v0')
obs = env.reset()[0]
while step < 2500:

    action = policy.predict(obs, deterministic=True)[0]
    #print(action)
    obs, _reward, done, *_, _info = env.step(action)
    #obs = obs
    env.mj_render()
    step += 1
    if done:
        obs = env.reset()[0]
        #break
        print("START")

env.close()