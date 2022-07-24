from distutils.log import error
from NSP_builder import NSPbuilder
from NSP_greedy import NSPgreedy
from job import Job
from apq import KeyedPQ
import numpy as np

from nsp_sandbox.Block_condition import Constraint

class Nurse():
    id: int
    open: KeyedPQ[Job]
    closed: KeyedPQ[Job]
    assigned: Job
    fixed: bool #wether the assignment can be changed or not
    blocking_register : dict[str,dict[int,Constraint]]
    assignment_hooks : dict[str,dict[int,Constraint]]
    def __init__(self,manager:NSPgreedy, problem: NSPbuilder, id: int, day: int,manage_options: bool = True) -> None:
        
        self.problem = problem
        self.manager = manager
        self.id = id
        self.str = str(id) + "_" + str(day)
        self.day = day
        self.open = KeyedPQ(max_heap=True)
        self.closed = KeyedPQ(max_heap=True)
        self.n_options = 0
        self.weights = self.problem.W[id]
        self.manage_options = manage_options
        self.setup_conditions()
        if self.manage_options:
            self.enlist_in_jobs()
            self.manager.log("Added Nurse {}".format(self.str))
    
    def add_option(self,job: Job):
        if not self.open.get(job.str) and self.problem.S[job.id,self.id]:
            self.open.add(job.str, self.score(job),job)
            self.n_options += 1
            self.update()
    
    
    def update(self):
        self.best_job = self.open.peek().data
        self.best_score = self.open.peek().value
        self.best_id = self.open.peek().key
    
    def score(self,job: Job):
        return self.weights[self.day,job.id]
    
    def setup_conditions(self):
        self.blocking_register = {}
        self.assignment_hooks = {}
    
    def add_hook(self,job_id:int,bc:Constraint):
        if not self.assignment_hooks[str(job_id) + "_" + str(self.day)]:
           self.assignment_hooks[str(job_id) + "_" + str(self.day)] = {bc.id: bc} 
        else: 
           self.assignment_hooks[str(job_id) + "_" + str(self.day)][bc.id] = bc 
    
    def remove_option(self,job: Job,fixed: bool = False):
        if self.open.get(job.str):
            self.open.__delitem__(job.str)
            if not fixed:
                self.closed.add(job.str,self.score(job),job)
            self.n_options -= 1
            self.update()
    
    def block(self,job: int, bc: Constraint):
        id_ = str(job) + "_" + str(self.day)
        if self.open.get(id_):
            self.open.get(id_).data.remove_option(self)
            self.remove_option(self.open.get(id_).data)
            self.blocking_register[id_] = {bc.id: bc}
        else:
            #add blocking condition to register for later
            self.blocking_register[id_][bc.id] = bc
        
    
    def unblock(self,job:int, bc: Constraint):
        id_ = str(job) + "_" + str(self.day)
        if id in self.blocking_register and bc.id in self.blocking_register[id_]:
            self.blocking_register[id_].__delitem__(bc.id)
            if not bool(self.blocking_register[id_]):
                self.closed.get(id_).data.re_add_option(self)
                self.re_add_option(self.closed.get(id_).data)
    
    def enlist_in_jobs(self):
        for job in self.problem.skill_lists[self.id]:
            j = self.manager.job_dict[job][self.day]
            if j.demand >0:
                self.add_option(j)
                j.add_option(self)

    def assing(self,job:Job, fixed:bool = False, shift_assign= False):
        self.fixed = fixed
        #should be called from job if nurse was free hence unassigend and the job should be open here
        if not self.assigned and self.open.get(job.str):
            if job.assign(self,fixed,shift_assign): #tell job to close itselfe/eg accept
                #job should now be closed
                self.set_taken(job) #block other possible assignments if other free nurses are available
                self.assigned = job #remember_assignment
                self.problem.start[self.id,self.day,job.id] = 1 #write to problem
                self.manager.log("Assigned Nurse {} to Job {} on Day {}".format(self.id,job.id,self.day))
            else:
                error("Tried blocked assignment!")
        else:
            error("Inconsistency")

    def is_assigned_to(self,job):
        return self.assigned == job
        
    def unassign(self):
        if self.assigned and not self.fixed:
            #first open job
            if self.assigned.unassign():
                #update constrains
                for key in self.assignment_hooks[self.assigned.str].keys():
                    self.assignment_hooks[self.assigned.str][key].drop_assignment((self.id,self.day,self.assigned.id))
                self.assigned = None
                #check if nurse is free for every job
                for job in self.open.ordered_iter():
                    if bool(self.blocking_register[job.data.str]):
                        job.data.re_add_option(self)
                for job in self.closed.ordered_iter():
                    if bool(self.blocking_register[job.data.str]):
                        job.data.re_add_option(self)
                return True
            return False
        else:
            return False

    def re_add_option(self,job: Job):
        if self.closed.get(job.str):
            if not bool(self.blocking_register[job.str]) and not job.assigned:
                self.closed.__delitem__(job.str)
                self.open.add(job.str,self.score(job),job)
                self.update()
                return True
            return False
        else:
            return False

    def set_taken(self,blocking_job:Job):
        #removes itselfe from all job options
        #one job constraint
        for job in self.open.ordered_iter():
            job.data.remove_option(self,blocking_job,self.fixed)
        for job in self.closed.ordered_iter():
            job.data.remove_option(self,blocking_job,self.fixed)  
        #remaining constrains
        for constraint_id in self.assignment_hooks[blocking_job.str].keys():
            self.assignment_hooks[blocking_job.str][constraint_id].new_assignment((self.id,self.day,blocking_job.id))

    def switch_to(self,job:Job):
        if not self.fixed:
            if self.open.get(job.str):
                #drop old solution column
                self.problem.start[self.id,self.day,self.assigned.id] = 0
                self.manager.log("Nurse {} needs to switch from {} to {} on day {}".format(self.id,self.assigned.id,job.id,self.day))
                self.assigned.unassign() #start cleanup from old job
                for constraint_id in self.assignment_hooks[self.assigned.str].keys():
                    self.assignment_hooks[self.assigned.str][constraint_id].drop_assignment((self.id,self.day,self.assigned.id))
                
                while bool(self.blocking_register[job.str]):
                    self.blocking_register[job.str][0].lift_for((self.id,self.day,job.id))#constraint will remove itselfe.
                #nurse still not open for other jobs
                self.assigned = job #remember new job
                for constraint_id in self.assignment_hooks[self.assigned.str].keys():
                    self.assignment_hooks[self.assigned.str][constraint_id].new_assignment((self.id,self.day,self.assigned.id))
                self.problem.start[self.id,self.day,self.assigned.id] = 1 
            else:
                error("Nurse switches to job that is not in her open list")
        return False

    