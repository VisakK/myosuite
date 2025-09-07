from stable_baselines3 import SAC
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
from myosuite.utils import gym
from stable_baselines3.common.policies import ActorCriticPolicy

wandb.init(project="myo_challenge", sync_tensorboard=True)
env = gym.make('myoChallengeSoccerP1-v0')
policy_name = "stand_policy_p1_SAC"
policy_kwargs = dict(activation_fn=nn.Tanh,
                     net_arch=dict(pi=[2048, 1024,512,1024], qf=[2048,1024, 512,1024]),)
                     #use_sde=True,
                     #use_expln=True) 

model = SAC("MlpPolicy", env,
            learning_rate=0.0002,
            policy_kwargs= policy_kwargs,
            verbose=1,
            use_sde=True,
            use_sde_at_warmup=True,
            tensorboard_log="./ppo_myo_challenge_soccer_p1_tensorboard/",
            policy_name="./policies/" + policy_name)

model.learn(total_timesteps=10000000, callback=WandbCallback())
model.save(f"ppo_myo_challenge_{policy_name}_2nd_run")
env.close()