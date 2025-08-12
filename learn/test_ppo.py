from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from myosuite.utils import gym

wandb.init(project="myo_challenge", sync_tensorboard=True)
env = gym.make('myoChallengeSoccerP1-v0')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_myo_challenge_soccer_p1_tensorboard/")
model.learn(total_timesteps=500000, callback=WandbCallback())
model.save("ppo_myo_challenge_soccer_p1")
env.close()