'''
Created on Nov 2, 2017

@author: silvai
'''
import gym
from gym import envs

def example():
    print(len(envs.registry.all()))
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        
        
example()