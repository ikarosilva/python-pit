'''
Created on Nov 2, 2017

@author: silvai
'''
import gym
from gym import envs


def basic_policy(obs):
    return 0 if obs[2]<0 else 1

def example():
    print(len(envs.registry.all()))
    env = gym.make('CartPole-v0')
    obs=env.reset()
    for _ in range(1000):
        a=basic_policy(obs)
        env.render()
        obs,rw,dn,info=env.step(a) # take a random action
        
        
example()