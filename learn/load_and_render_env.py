from myosuite.utils import gym
# Soccer environment - myoChallengeSoccerP1-v0
env = gym.make('myoChallengeSoccerP1-v0')
env.reset()
for _ in range(2000):
    env.mj_render()
    env.step(env.action_space.sample()*0.0) # take a random action
env.close()