'''
Created on Nov 2, 2017

@author: silvai
'''
from gym import Env
import numpy as np

class MarkovDecisionProcess(Env):
    def __init__(self,T,R):
        self.T=T
        self.R=R
        
    def _step(self, action): 
        raise NotImplementedError
    
    def _reset(self): 
        raise NotImplementedError
    
    def _render(self, mode='human', close=False): 
        return
    
    def _seed(self, seed=None): 
        return []
        