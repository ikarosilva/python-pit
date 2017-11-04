'''
Created on Nov 2, 2017

@author: silvai
'''
import gym
import tensorflow as tf 
from tensorflow.contrib.layers import fully_connected
import numpy as np 

env = gym.make('CartPole-v0')

n_inputs=4
n_hidden=4
n_outputs=1
initializer = tf.contrib.layers.variance_scaling_initializer()
learning_rate=0.01

X=tf.placeholder(tf.float32,shape=[None, n_inputs],name='X')
hidden = fully_connected(X, n_hidden,activation_fn=tf.nn.elu,
                         weights_initializer=initializer)
logits = fully_connected(hidden,n_outputs,activation_fn=None,
                         weights_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1,values=[outputs, 1-outputs],name='out')
tmp_act=tf.log(p_left_and_right,name='tmp_act')
action = tf.multinomial(tmp_act,num_samples=1,name='action')
tf.add_to_collection('policy', action)

y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [ grad for grad, variable in grads_and_vars]
gradient_placeholders =[]
grads_and_vars_feed=[]
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder,variable))
training_op= optimizer.apply_gradients(grads_and_vars_feed)
init = tf.global_variables_initializer()
saver=tf.train.Saver()
       
n_iterations=250
n_max_steps = 1000
n_games_per_update=10
save_iterations = 10
discount_rate=0.95


def discount_rewards(rewards,discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards=0
    for step in reversed(range(len(rewards))):
        cumulative_rewards =rewards[step] + cumulative_rewards*discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards 

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards,discount_rate) for rewards in all_rewards]
    flat_rewards =np.concatenate(all_discounted_rewards)
    reward_mean=flat_rewards.mean()
    reward_std=flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


def basic_policy(obs):
    return 0 if obs[2]<0 else 1

def example1():
    obs=env.reset()
    for _ in range(n_iterations):
        a=basic_policy(obs)
        env.render()
        obs,rw,dn,info=env.step(a) # take a random action

def train():
    with tf.Session() as sess:
        init.run()  
        for iteration in range(n_iterations):
            all_rewards=[]
            all_gradients=[]
            for game in range(n_games_per_update):
                current_rewards=[]
                current_gradients=[]
                obs = env.reset()
                for step in range(n_max_steps):
                    action_val, gradients_val = sess.run(
                        [action,gradients], 
                        feed_dict={X:obs.reshape(1,n_inputs)})
                    #env.render()
                    obs,reward,done,info=env.step(action_val[0][0])
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
                
                
            all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)
            feed_dict={}
            for var_index, grad_placeholder in enumerate(gradient_placeholders):
                mean_gradients = np.mean(
                    [reward*all_gradients[game_index][step][var_index]
                        for game_index, rewards in enumerate(all_rewards)
                        for step, reward in enumerate(rewards)],
                                          axis=0)
                
                feed_dict[grad_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            if iteration % save_iterations ==0:
                print('Saving model (%s / %s)...'%(iteration,n_iterations))
                saver.save(sess,"./my_policy_net_pg.ckpt")
    print('Learning completed! Deploying...')

def deploy():
    with tf.Session(config=tf.ConfigProto(log_device_placement=True,device_count = {'GPU': 0})) as sess:
        init.run()
        saver.restore(sess,"./my_policy_net_pg.ckpt")
        print("Model restored.")
        for _ in range(n_iterations):
            for _ in range(n_games_per_update):
                obs = env.reset()
                for _ in range(n_max_steps):
                    action_val, _ = sess.run(
                        [action,gradients], 
                        feed_dict={X:obs.reshape(1,n_inputs)})
                    obs,_,done,_=env.step(action_val[0][0])
                    if done:
                        break
                    
train()
print('Done!')