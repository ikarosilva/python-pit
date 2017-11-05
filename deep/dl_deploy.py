import pyglet
import gym
import tensorflow as tf
from time import sleep

env = gym.make('CartPole-v0')
env.render()
    
n_iterations=100
n_games_per_update=10
n_max_steps = 1000

def basic_policy(obs):
    return 0 if obs[2]<0 else 1

def deploy(env):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True,device_count = {'GPU': 1})) as sess:
        saver = tf.train.import_meta_graph("./my_policy_net_pg.ckpt.meta")
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        #for op in graph.get_operations():
        #    print str(op.name)
        X = graph.get_tensor_by_name("X:0")
        action =graph.get_tensor_by_name("action/Multinomial:0")
        n_inputs=4
        #deploy0(env)
        obs = env.reset()
        for iteration in range(600000):
            action_val= sess.run(action,feed_dict={X:obs.reshape(1,n_inputs)})
            env.render()
            obs,_,done,_=env.step(action_val[0][0])
            if done:
                print("Done with simulation step:%s"%(iteration))
                break
   
def deploy0(env):
    obs=env.reset()
    for _ in range(300):
        a=basic_policy(obs)
        print a
        env.render()
        obs,rw,dn,info=env.step(a) # take a random action
        
deploy(env)
    
    