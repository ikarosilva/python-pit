'''
Created on Nov 2, 2017

@author: silvai
'''
from gym import Env
import numpy as np
import numpy.random as rnd

class MarkovDecisionProcess(Env):
    def __init__(self,T,R,possible_actions):
        self.T=T
        self.R=R
        s,a,_=T.shape
        self.number_of_states=s
        self.number_of_actions=s
        self.possible_actions=possible_actions
        
    def _step(self, action): 
        raise NotImplementedError
    
    def _reset(self): 
        raise NotImplementedError
    
    def _render(self, mode='human', close=False): 
        return
    
    def _seed(self, seed=None): 
        return []

def MDP_1():
    T=np.array([#s a s'
                [[0.7, 0.3, 0.],[1., 0., 0.],[0.8, 0.2, 0.]],
                [[0., 1., 0.],[np.nan, np.nan, np.nan],[0., 0., 1.]],
                [[np.nan, np.nan, np.nan],[0.8, 0.1, 0.1],[np.nan, np.nan, np.nan]],
                ])
    R=np.array([#s a s'
                [[10., 0., 0.],[0., 0., 0.],[0., 0., 0.]],
                [[10., 0., 0.],[np.nan, np.nan, np.nan],[0., 0., -50.]],
                [[np.nan, np.nan, np.nan],[40., 0., 0.],[np.nan, np.nan, np.nan]],
                ])
    possible_actions=[[0,1,2],[0,2],[1]]
    return MarkovDecisionProcess(T,R,possible_actions)

def Q_value_iteration(env,discount_rate=0.95,n_iterations=100):
    Q=np.full((env.number_of_states,env.number_of_actions),-np.inf)
    for state, actions in enumerate(env.possible_actions):
        Q[state,actions]=0.0
    for _ in range(n_iterations):
        Q_prev=Q.copy()
        for s in range(env.number_of_states):
            for a in env.possible_actions[s]:
                Q[s,a]= np.sum([
                                env.T[s,a,sp]*(env.R[s,a,sp] + discount_rate*np.max(Q_prev[sp]))
                                for sp in range(3)
                    ])
    return Q

def Q_value_learning(env,discount_rate=0.95,learning_rate0=0.05,learning_rate_decay=0.1,n_iterations=20000):
    s=0
    Q=np.full((env.number_of_states,env.number_of_actions),-np.inf)
    for state, actions in enumerate(env.possible_actions):
        Q[state,actions]=0.0
    for iteration in range(n_iterations):
        a =rnd.choice(env.possible_actions[s])
        sp = rnd.choice(range(env.number_of_states),p=env.T[s,a])
        reward=env.R[s,a,sp]
        learning_rate =learning_rate0 / (1+ iteration*learning_rate_decay)
        Q[s,a]= learning_rate*Q[s,a] + (1-learning_rate)*(reward + discount_rate*np.max(Q[sp]))
        s=sp
    return Q
            
if __name__ == '__main__':
    ex1=MDP_1()
    Q=Q_value_iteration(ex1,discount_rate=0.95)
    print Q 
    print np.argmax(Q,axis=1)   
    
    Q=Q_value_learning(ex1,discount_rate=0.95)
    print Q 
    print np.argmax(Q,axis=1) 
    
    
    