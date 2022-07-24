from distutils.log import error
from typing import Tuple
from NSP_builder import NSPbuilder
from NSP_greedy import NSPgreedy
from nurse import Nurse
from job import Job
import numpy as np

class Constraint():
    manager: NSPgreedy
    mask: list # list of indexes with (day,nurse,job)
    blocking_positions: list
    def __init__(self,manager,n_states,min,mask:list,weights,bound,to_block,name):
        self.id = name
        self.to_block = to_block
        self.name = name
        self.manager = manager
        self.min = min
        self.index_pos = {}
        self.pos_index = {}
        i = 0
        self.mask = mask
        for index in mask:
            self.index_pos[index] = i
            self.pos_index[i] = index
            i += 1
        self.weights = weights
        self.bound = bound
        self.state = np.zeros(n_states)
        self.n_states = n_states
        self.res = 0
        self.blocking_positions = []
        self.setup_hooks()
    
    def setup_hooks(self):
        for index in self.mask:
            if self.manager.nurse_dict[index[0]][index[1]]:
                self.manager.nurse_dict[index[0]][index[1]].add_hook(index[2],self)

            


    def new_assignment(self,index):
        if not self.state[self.index_pos[index[0]][index[1]][index[2]]]:
            self.state[self.index_pos[index[0]][index[1]][index[2]]] = 1
            self.res += self.weights[self.index_pos[index[0]][index[1]][index[2]]]
            self.blocking_positions.append(index)
            self.check()
    
    def drop_assignment(self,index):
        if self.state[self.index_pos[index[0]][index[1]][index[2]]]:
            self.state[self.index_pos[index[0]][index[1]][index[2]]] = 0
            self.res -= self.weights[self.index_pos[index[0]][index[1]][index[2]]]
            self.blocking_positions.remove(index)
            self.check()
    
    def check(self):
        if self.res==self.bound:
                self.block()
        else:
            if self.min:
                if self.res > self.bound:
                    self.unblock()

            else:
                if self.res < self.bound:
                    self.unblock()
    
    def block(self):
        self.blocking = True
        for index in self.to_block:
            self.manager.nurse_dict[index[1]][index[0]].block_job(index[2],self)

    def lift_for(self, index: Tuple):    
            while self.blocking:
                #undo last assignment that changed this condition
                position = self.blocking_positions[-1]
                self.manager.nurse_dict[position[0]][position[1]].unassign()

    def unblock(self):
        self.blocking = False
        for index in self.to_block:
            self.manager.nurse_dict[index[1]][index[0]].unblock_job(index[2],self)

        
