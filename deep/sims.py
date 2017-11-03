'''
Created on Nov 2, 2017

@author: silvai
'''
from gym import Env
import numpy as np
import numpy.random as rnd
from dateutil import zoneinfo

class MarkovDecisionProcess(Env):
    def __init__(self,T,R,possible_actions):
        self.T=T
        self.R=R
        s,a,_=T.shape
        self.number_of_states=s
        self.number_of_actions=a
        self.possible_actions=possible_actions
        self.s=0
        
    def _step(self, action):  
        done=False
        sp = rnd.choice(range(self.number_of_states),p=self.T[self.s,action])
        reward=self.R[self.s,action,sp]
        self.s=sp
        info=''
        return self.s,reward,done,info
    
    def _reset(self): 
        self.s=0
    
    def _render(self, mode='human', close=False): 
        return
    
    def _seed(self, seed=None): 
        return []
    
    def sample(self):
        return rnd.choice(self.possible_actions[self.s])
        
    def Q_value_iteration(self,discount_rate=0.95,n_iterations=100):
        Q=np.full((self.number_of_states,self.number_of_actions),-np.inf)
        for state, actions in enumerate(self.possible_actions):
            Q[state,actions]=0.0
        for _ in range(n_iterations):
            Q_prev=Q.copy()
            for s in range(self.number_of_states):
                for a in self.possible_actions[s]:
                    Q[s,a]= np.sum([
                                    self.T[s,a,sp]*(self.R[s,a,sp] + discount_rate*np.max(Q_prev[sp]))
                                    for sp in range(3)
                        ])
        return Q

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


def Q_value_learning(env,discount_rate=0.95,learning_rate0=0.05,learning_rate_decay=0.1,n_iterations=20000):
    s=env.reset()
    Q=np.full((env.number_of_states,env.number_of_actions),-np.inf)
    for state, actions in enumerate(env.possible_actions):
        Q[state,actions]=0.0
    for iteration in range(n_iterations):
        a = env.sample()
        sp,reward,_,_ = env.step(a)
        learning_rate =learning_rate0 / (1+ iteration*learning_rate_decay)
        Q[s,a]= learning_rate*Q[s,a] + (1-learning_rate)*(reward + discount_rate*np.max(Q[sp]))
        s=sp
    return Q
            
if __name__ == '__main__':
    ex1=MDP_1()
    Q=ex1.Q_value_iteration(discount_rate=0.95)
    print Q 
    print np.argmax(Q,axis=1)   
    
    Q=Q_value_learning(ex1,discount_rate=0.95)
    print Q 
    print np.argmax(Q,axis=1) 
    
    
    