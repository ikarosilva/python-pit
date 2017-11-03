'''
Created on Nov 2, 2017

@author: silvai
'''
import gym
from gym import envs
import tensorflow as tf 
from tensorflow.contrib.layers import fully_connected

n_inputs=4
n_hidden=4
n_outputs=1
initializer = tf.contrib.layers.variance_scaling_initializer()
learning_rate=0.01

X=tf.placeholder(tf.float32,shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden,activation_fn=tf.nn.elu,
                         weights_initializer=initializer)
logits = fully_connected(hidden,n_outputs,activation_fn=None,
                         weights_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1,values=[outputs, 1-outputs])
action = tf.multinomial(tf.log(p_left_and_right),num_samples=1)

def basic_policy(obs):
    return 0 if obs[2]<0 else 1

def example1():
    print(len(envs.registry.all()))
    env = gym.make('CartPole-v0')
    obs=env.reset()
    for _ in range(1000):
        a=basic_policy(obs)
        env.render()
        obs,rw,dn,info=env.step(a) # take a random action
        
        
example1()