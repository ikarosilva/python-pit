'''
Created on Feb 22, 2020

@author: ikaro
'''
from collections import OrderedDict
import numpy as np

class BinaryTree():
    '''
        Binary tree implementation via Huffman coding
    '''
    def __init__(self, tree):
        '''
            Intialize tree, using following format;
            {'0':val_0,'00':,val_00,'01':val_01....,etc}
            For most significant digit, '0' is left, '1' is right relative to parent (second most significant digit)
        '''
        tree=OrderedDict(sorted(tree.items(), key=lambda item: int(item[0],2)))
        self.tree=tree
        
    def breadth_traverse(self,function,reverse=False):
        nodes=list(self.tree.keys())
        nodes= nodes[::-1] if reverse else nodes
        for node in nodes:
                function(node, self.tree[node])
    
    def is_leaf(self,node):
        return (node+"0" not in self.tree) and (node+"1" not in self.tree)
    
    def navigate(self,node):
        left=node+"0" if node+"0" in self.tree else None
        right=node+"1" if node+"1" in self.tree else None
        up=node[:-1] if len(node)>1 else None
        is_leaf=self.is_leaf(node)
        return left,right,up,is_leaf
    
    def depth_traverse(self,function):
        '''
        (1->2) : 1 
        (10->5) : 2 
        (100->9) : 4 
        (101->10) : 5
        (1010->21) : 7 
        (1011 ->22) : 8 
        (11 -> 6) : 3 
        (111 -> 14) : 6 
        (1111 -> 28) : 9 
        '''
        node="1"
        nodes=list(self.tree.keys())
        passed=[]
        while len(nodes):
            left,right,up,is_leaf=self.navigate(node)
            if(node in nodes and node not in passed):
                function(node,self.tree[node])
                passed.append(node)
            if(is_leaf):
                del self.tree[node]
                nodes.remove(node)
                node=up
            elif(left):
                node=left
            elif(right):
                node=right
            elif(up is None):
                node=right

            
    def number_of_levels(self):
        return np.max([len(x) for x in self.tree.keys()])+1
    
    def pretty_print(self):
        N=self.number_of_levels()
        buffer=int(2**N)
        level=0
        level_str=[""]
        step=0
        count=0
        for ind in range(2**N):
            node='{0:b}'.format(ind+1)
            this_level=len(node)
            if(this_level != level):
                #Start new level, print old buffer and reset
                print("".join(level_str))
                level_str=list(" "*buffer)
                level=this_level
                step=int(buffer/(2**level))
                count=1
            if node in self.tree:
                left,right,_,_=self.navigate(node)
                branch_pad=int(step/2)-1
                if(left):
                    level_str[int(step*count)-1-branch_pad:int(step*count)-1]="/"+"-"*branch_pad
                if(right):
                    level_str[int(step*count)+1:int(step*count)+1+branch_pad]="-"*branch_pad+"\\"
                level_str[int(step*count)]=str(self.tree[node])
            count = count + 2
        print("".join(level_str))
        
if __name__ == '__main__':  
    '''
            1
       /       \
      2          3
    /  \          \
   4    5          6
      /   \      /   \
     7     8    11    9
                      \
                       10
    '''
    print("Running Tree Example...")
    tree=BinaryTree({'1':1,'10':2,'11':3,'100':4,'101':5,'111':6,'1010':7,'1011':8,'1111':9,'1110':11,'11111':10})
    tree.pretty_print()
    print("Done")