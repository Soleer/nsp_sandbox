from distutils.log import error
from __future__ import annotations
from NSP_builder import NSPbuilder
from NSP_greedy import NSPgreedy
from Block_condition import BlockCondition
import numpy as np
from nurse import Nurse
from apq import KeyedPQ


class Job():
    id: int
    free: KeyedPQ[Nurse]
    taken: KeyedPQ[Nurse]
    n_options: int
    manager: NSPgreedy
    assigned: Nurse
    fixed: bool #wether the assignment can be changed or not

    def __init__(self,manager: NSPgreedy,problem: NSPbuilder, id: int, day: int,managed: bool = True) -> None:
        self.second_chance_counter = 0
        self.id = id
        self.str = str(id)+"_"+str(day)
        self.problem = problem
        self.manager = manager
        self.day = day
        self.demand = self.problem.demand[self.day,self.id] #currently only 1 0 demand possible => assigned var must be changed
        self.type = self.problem.job_types[self.id]
        self.standort = self.problem.job_standorte[self.id]
        self.shift_job = self.type == 1 #for now
        
        self.n_options = 0
        self.position = -self.demand
        self.free = KeyedPQ(max_heap=True)
        self.taken = KeyedPQ(max_heap=True)
        if self.demand > 0 and managed:
            if self.shift_job:
                self.det_shift()
            self.manager.job_open.add(self.str,self.position,self) #add to open jobs
            self.manager.log("Added Job {}".format(self.str))

    
    def add_option(self,nurse: Nurse):
        if not self.free.get(nurse.str) and self.problem.S[self.id,nurse.id]:
            if not self.taken.get(nurse.str):
                self.free.add(nurse.str,nurse.score(self.id),nurse)
                self.update()
                self.increase_options(1)
            else:
                self.re_add_option(nurse)

    def update(self):
        self.best_value = self.free.peek().value
        self.best_key = self.free.peek().key
        self.best_nurse = self.free.peek().data
    
    def remove_option(self,n:Nurse,fixed: bool = False):
        if self.free.get(n.str):
            self.free.__delitem__(n.str)
            if not fixed: #remember option for possible steal
                self.taken.add(n.str,n.score(self.id),n)
            self.update()
            self.increase_options(-1)
            return True
        return False
    
    def det_shift(self):
        for s in self.problem.shifts[self.id]:
            if self.day in s:
                self.shift = s
                return
        self.shift = []
        self.shift_job = False
        

    
    def increase_options(self,s:int):
        self.n_options += s
        self.position += s
        self.manager.job_open.add_or_change_value(self.str,self.position,self)
    
    def assign(self,nurse:Nurse,fixed: bool = False, shift_assign=False):
        #gets called from nurse
        if not self.assigned:# should be true-> we do not allow switching nurse from job side e.g. first reassign nurse then job
            if self.free.get(nurse.str):
                self.fixed = fixed
                self.set_closed() #close this job 
                self.assigned = nurse #remember assignment from this side
                self.manager.job_open.__delitem__(self.str)#removes itselfe from manager list 
                if self.shift_job and not shift_assign:
                    #assign to other shiftdays as well
                    for d in self.shift:
                        if d != self.day:
                            self.manager.nurse_dict[nurse.id][d].assign(self.manager.job_dict[self.id][d],fixed=False,shift_assign = True)
                return True
            else:
                return False
        else: 
            return self.assigned == nurse
        
    def set_closed(self):
        for nurse in self.free.ordered_iter():
            nurse.data.remove_option(self,self.fixed)
        for nurse in self.taken.ordered_iter():
            nurse.data.remove_option(self,self.fixed)
    
    def unassign(self, shift_unassign = False):
        #gets called if nurse needs to switch
        if self.assigned and not self.fixed: #should be true
            self.assigned = None
            #readds this job to all nurses, where it is not blocked by something else than the demand
            for nurse in self.free.ordered_iter():
                nurse.data.re_add_option(self)
            for nurse in self.taken.ordered_iter():
                nurse.data.re_add_option(self)
                if self.shift_job and not shift_unassign:
                   for d in self.shift:
                        if d != self.day:
                            self.manager[self.id][d].unassign(True) 
            self.manager.job_open.add(self.str,self.n_options,self)
            return True
        else:
            return False

    def re_add_option(self,nurse:Nurse, shift_re_add=False):
        if self.taken.get(nurse.str):
            self.taken.__delitem__(nurse.str)
            self.free.add(nurse.str,nurse.score(self),nurse)
            self.increase_options(1)
            return True
        else:
            return False
    
    def assign_best(self):
        assigned = True
        if self.demand >0: # if called by manager demand should be 1 since it was in job_open
            assigned = False
            if self.n_options >0: #number of available nurses to this job
                nurse = self.free.peek().data #best score 
                nurse.assign(self) #assign best
                assigned = True #done
            else:
                c = 0 #all allowed nurses are already assigned hence we need to steal one
                #iterate over taken nurses ordered by there score
                for nurse in self.taken.ordered_iter():
                    if c == self.second_chance_counter: #assign first not already tried nurse to this job
                        self.second_chance_counter += 1
                        if nurse.data.switch_to(self):#if successful stop iteration
                            assigned = True
                            break
                        #else go to next (e.g blocking job can not be reassigned (Urlaub etc.))
                    c +=1
        else:
            error("job with no demand gets assigned")
        return assigned

                    