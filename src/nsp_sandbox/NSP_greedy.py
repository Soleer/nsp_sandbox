from distutils.log import error
from pyrsistent import s

from nsp_sandbox.Block_condition import Constraint
from .NSP_builder import NSPbuilder
from apq import KeyedPQ
from job import Job
import numpy as np
from nurse import Nurse

class NSPgreedy():

    job_open: KeyedPQ[Job]
    job_dict: dict[int,dict[int,Job]]
    nurse_dict: dict[int,dict[int,Nurse]]
    log_list :list[dict] = []
    extended: bool #extended greedy?

    def __init__(self,problem: NSPbuilder, extended: bool = False):
        self.extended = extended
        self.problem = problem
        self.logging = self.problem.print_info
        self.cost = 0
        self.time = 0
        #gens jobregister
        self.init_iterator()
        #gens nurses and adds connects everything
        self.init_nurses()
        #add before and after and current already planned
        self.gen_fixed_entries()
        #init constraints
        self.init_constraints()
        #enter fixed entries
        self.assign_fixed_entries()

    
    def init_iterator(self):
        self.job_open = KeyedPQ()
        self.job_dict = {}
        for job in range(self.problem.n_jobs):
            self.job_dict[job] = {}
            for d in range(self.problem.n_days):
                self.job_dict[job][d] = Job(self,self.problem, job,d) #adds itselfe to job_open if necessary
    
    def init_nurses(self):
        for n in range(self.problem.n_nurses):
            for d in range(self.problem.n_days):
                self.nurse_dict[n][d] = Nurse(self,self.problem,n,d)
    
    def gen_fixed_entries(self):
        for n in range(self.problem.n_nurses):
            for d in range(self.problem.before.shape[1]):
                jobs = np.argwhere(self.problem.before[n,d,:]==1)
                if jobs:
                    self.job_dict[jobs[0]][-d] = Job(self,self.problem,jobs[0],self.problem.before.shape[1]-d,False)
                    self.nurse_dict[n][-d] = Nurse(self,self,self.problem,n,-d,False)
            for d in range(self.problem.after.shape[1]):
                jobs = np.argwhere(self.problem.before[n,d,:]==1)
                if jobs:
                    self.job_dict[jobs[0]][self.problem.n_days + d] = Job(self,self.problem,jobs[0],self.problem.n_days + d,False)
                    self.nurse_dict[n][self.problem.n_days + d] = Nurse(self,self.problem,n,self.problem.n_days + d,False)
    
    def assign_fixed_entries(self):
        for n in range(self.problem.n_nurses):
            for d in range(self.problem.n_days):
                jobs = np.argwhere(self.problem.pre_X[n,d,:]==1)
                if jobs:
                    j = jobs[0]
                    self.nurse_dict[n][d].assing(self.job_dict[j][d],True)
            for d in range(self.problem.before.shape[1]):
                jobs = np.argwhere(self.problem.before[n,d,:]==1)
                if jobs:
                    self.nurse_dict[n][-d].assign(self.job_dict[jobs[0]][-d],True) 
            for d in range(self.problem.after.shape[1]):
                jobs = np.argwhere(self.problem.before[n,d,:]==1)
                if jobs:
                    self.nurse_dict[n][self.problem.n_days + d].assign(self.job_dict[jobs[0]][self.problem.n_days + d],True)

    def init_constrains(self): 
        self.constraints = {}             
        #read constrains from problem... need good interface
        #job demand is one or zero and already managed by the job class -> blocks new assignments
        #job per day is set to one and already managed by the nurse class -> blocks new assignments
        #job demand is the condition for termination
        #assignment is only possible if skill constraint is fullfilled
        #preplan will be assigned afterwards and will be fixed
        #init shift_constrains:
        self.init_shift_constrains()
        self.init_break_days()
    
    def init_break_days():

    def init_shift_constrains(self):
        #for every shift job
        for j in np.argwhere(self.problem.job_types == 1):
            nurses = np.argwhere(self.problem.S[j,:] == 1)
            #for every shift
            si = 0
            for s in self.problem.shifts[j]:
                shiftdays = np.argwhere(s == 1)
                #for every nurse with skill
                for n in nurses:
                    mask = []
                    for d in shiftdays:
                        for jm in self.problem.skill_lists[n]:
                            if jm != j:
                                mask.append((n,d,jm))
                    n_states = len(mask)
                    c_name = "shift_{}_{}_{}".format(j,si,n) 
                    self.constraints[c_name] = Constraint(self,n_states,False,mask,np.ones(n_states),1,[(n,dm,j) for dm in shiftdays],c_name)
                si += 1


        
    
    def next_job(self):
        return self.job_open.peek().data
    
    def log(self,msg,status="Debug"):
        if status != "Debug" or self.logging:
            self.log_list.append({"status":status, "msg": msg})
    
    def solve(self):
        while self.next_job():
            job = self.next_job()
            if not job.assign_best():
                error("Could not find plan with assignment for {}" %(job.id))



