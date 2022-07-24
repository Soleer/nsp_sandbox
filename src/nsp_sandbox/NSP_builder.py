import random
import numpy as np
import math
from scipy.stats import bernoulli
from collections.abc import Callable
import gurobipy as gp

from scipy import signal


class NSPbuilder:
    print_info: bool
    version: str
    n_nurses: int
    n_jobs: int
    n_days: int
    n_weeks: int
    D: np.ndarray #demand (n_jobs,n_days)
    S: np.ndarray #skills (n_jobs,n_nurses)
    skill_lists: dict[int,list] #(nurse,list of job ids )
    W: np.ndarray #wishs (n_nurses,n_days,n_jobs)
    R: np.ndarray #rotations (n_nurses,n_days,n_jobs)
    prefs: np.ndarray # (n_nurses,n_jobs)
    pre_X: np.ndarray #(n_nurses,n_days,n_jobs)
    shifts: dict[int,list]
    model: gp.Model
    X: gp.MVar
        

    skill_exp_exp=0.30
    skill_exp_std=0.35
    skill_cov_exp=0.017
    skill_cov_std=0.03
    prob_rot_whole_window = 0.6
    pref_dist = [0.15,0.1,0.4,0.2,0.15]
    prob_kein_dienst_wish = 6/30
    prob_normal_dienst_wish = 1/7
    no_skill_requirement_precentage = 0.1
    default_padding= 14 # days before and after the window
    nurse_params: list[dict]
    kernels: list[dict]

    job_type_name = {0: "normal",1: "shift", 2: "night", 3: "long", 4: "versetzt"}

    pref_weight = 1
    wish_weight = 3
    rotation_weight = 2
    default_standortwechsel_weight = -2

    current_solution: np.ndarray
    before = np.array([])
    after= np.array([])


    constraints: list[Callable[[],bool]]



    def __init__(self,n_nurses,n_weeks,n_jobs, print_info=True) -> None:
        self.print_info = print_info
        self.version = "0.0.1"
        
        self.set_n_nurses(n_nurses)
        self.set_n_weeks(n_weeks)
        self.set_n_jobs(n_jobs)
        self.before = np.zeros((self.n_nurses,self.n_days,self.n_jobs))
        self.after= np.zeros((self.n_nurses,self.n_days,self.n_jobs))
    
    def create(self):
        self.model = gp.Model(name="nsp")
        self.X = self.model.addMVar((self.n_nurses,self.n_days,self.n_jobs),vtype=gp.GRB.BINARY,name="X")
        self.init_constrains()
        self.init_kernels()

    
    def init_constrains(self):
        self.constraints = []
        self.constraints.append(self.check_one_job_per_day_per_nurse)
        self.constraints.append(self.check_solution_meets_job_demand_per_day)
        self.constraints.append(self.check_skill_requirement_per_job)
        self.constraints.append(self.check_pre_plan)
        self.constraints.append(self.check_shifts)
        self.constraints.append(self.check_kernels)
        self.constraints.append(self.check_break_all_ten_days)
    
    def init_kernels(self):
        self.hard_kernels = []
        self.soft_kernels = []
        self.add_only_normal_after_night_kernel()
        self.add_type_limit_kernel(5,3)
        self.add_type_limit_kernel(7,4)
        self.add_reduce_standortwechsel_kernel()

    def add_reduce_standortwechsel_kernel(self,weight =0):
        if weight ==0:
            weight = self.default_standortwechsel_weight 
        kernel = {"padding": 0, "weight": weight, "offset": 1}
        core = np.zeros((2,self.n_jobs))
        for s in range(1,self.n_standorte):
            core[0,self.job_standorte == s] = 1/2 # all jobs in standort s
            core[1,self.job_standorte != s] = 1/2 # all jobs not in standort s
            core[1,self.job_standorte == 0] = 0 # dont count standort 0 e.g. kein standort
            kernel["kernel"] = core
            kernel["name"] = "standortwechsel_weg_von_"+str(s)
            self.soft_kernels.append(kernel)
        
    def add_type_limit_kernel(self,window,condition):
        kernel = {"padding": window//2, "condition": condition}
        count_types = np.zeros(self.n_jobs)
        count_types[self.job_types != 0] = 1 # all non normal
        count_types[self.job_types == 1] = 0 # dont count shift jobs
        core = np.tile(count_types,(window,1))
        kernel["kernel"] = core
        kernel["name"] = "type_limit_"+str(window)+"_"+str(condition)
        self.hard_kernels.append(kernel)
        
           
    
    def create_check_window(self,window,condition):
        def check_window(x):
            return (signal.convolve(x,np.ones(window),mode="valid") <= condition).all()
        return check_window
    
    def add_only_normal_after_night_kernel(self):
        kernel = {"name": "only_normal_after_night","padding": 0, "condition": 1}
        core = np.zeros((2,self.n_jobs))
        core[0,self.job_types == 2] = 1 # all night jobs
        core[1,self.job_types != 0] = 1 # all non normal
        kernel["kernel"] = core 
        self.hard_kernels.append(kernel)
    
    
    def set_n_nurses(self, n:int) -> None:
        self.n_nurses = n
        self.nurse_params = [{}]*n 
    
    def set_n_jobs(self, n:int) -> None:
        self.n_jobs = n +2 # +2 for the holiday job 0 and break job 1
        self.n_standorte = max(math.ceil(n/10) ,2) + 1# +1 for no standort (0)
        self.n_job_types = min(math.ceil(n/10),3) + 2 # +1 for normal job (0) +1 for shift job (1) 2 => night, 3 => long, 4 => versetzt
        self.n_no_skill_jobs = math.ceil(n*self.no_skill_requirement_precentage)
        self.job_types = np.array(random.choices(range(self.n_job_types),weights= [1]*self.n_job_types,k=self.n_jobs))
        self.job_types[0] = 0
        self.job_types[1] = 0

        self.wish_dist  = [1- self.prob_kein_dienst_wish - 1./(7.*(self.n_job_types-1))]+ [self.prob_kein_dienst_wish] + [self.prob_normal_dienst_wish /(self.n_job_types -1)] * (self.n_job_types - 1) # nothing, k, normal wish
        self.job_standorte = np.array(random.choices(range(self.n_standorte),weights= [1]*self.n_standorte,k=self.n_jobs))
        self.job_standorte[0] = 0
        self.job_standorte[1] = 0

    def set_n_weeks(self, n:int) -> None:
        self.n_weeks = n
        self.n_days = n * 7

    def gen_skills(self):
        self.S = np.zeros((self.n_jobs,self.n_nurses))
        self.gen_skill_cov()
        self.job_skill_exp = np.random.normal(loc=self.skill_exp_exp,scale=self.skill_exp_std,size=self.n_jobs)
        self.job_skill_exp[self.job_skill_exp <0.01] = 0.01
        self.job_skill_exp[self.job_skill_exp >=1] = 1
        self.job_skill_prob = np.random.multivariate_normal(mean=self.job_skill_exp,cov=self.job_skill_cov,size=self.n_nurses)
        for n in range(self.n_nurses):
            self.job_skill_prob[n][ self.job_skill_prob[n] < 0.01] = 0.01
            self.job_skill_prob[n][ self.job_skill_prob[n] >= 1] = 1
            self.S[:,n] = np.random.binomial(1,self.job_skill_prob[n])
        self.S[:self.n_no_skill_jobs +2,:] = 1 # +2 for the holiday job 0 and break job 1
        self.setup_skill_lists()
    
    def setup_skill_lists(self):
        self.skill_lists = {}
        for n in range(self.S.shape[1]):
            self.skill_lists[n] = np.argwhere(self.S[:,n])

    def gen_skill_cov(self):
        self.top_cov = np.random.normal(loc=self.skill_cov_exp,scale=self.skill_cov_std,size=int(self.n_jobs*(self.n_jobs+1)/2.))
        self.job_skill_cov = np.zeros((self.n_jobs,self.n_jobs))
        k = 0
        for i in range(self.n_jobs):
            self.job_skill_cov[i,i] = self.top_cov[k]
            k +=1
            for j in range(i+1,self.n_jobs):
                s = self.top_cov[k]
                self.job_skill_cov[i,j] = s
                self.job_skill_cov[j,i] = s
                k += 1

    def gen_shift_patterns(self):
        self.shifts = {}
        for j in np.argwhere(self.job_types == 1):
                #generate random shift plus rewrite Demand to match
                self.shifts[j] = self.gen_shift_set(j)

    def gen_shift_set(self, job_id:int) -> list:
        shift_set = []
        self.D[job_id,:] = 0
        shift_pattern = np.argwhere(bernoulli.rvs(4/7,size=7)==1)
        for i in range(self.n_weeks): #every week the same shift pattern
            week_pattern = shift_pattern + i*7
            self.D[job_id,week_pattern] = 1
            shift_set.append(week_pattern)
        return shift_set

    def gen_demand(self):
        self.D = np.zeros((self.n_jobs,self.n_days),)
        for j in range(self.n_jobs):
            self.D[j] = self.gen_demandset(j)
    
    def gen_demandset(self, j,type="random"):
        week = bernoulli.rvs(5./7.,size=7)
        if type == "week":
            week = [1,1,1,1,1,0,0]
        if type == "weekend":
            week = [0,0,0,0,1,1,1]
        if type == "everyday":
            week = [1,1,1,1,1,1,1]
        demand = np.tile(week, self.n_weeks)
        return demand
    
    def gen_prefs(self):
        self.prefs = np.zeros((self.n_nurses,self.n_jobs))
        for n in range(self.n_nurses):
            self.prefs[n] = self.gen_prefset(n)
        
    def gen_prefset(self, n,type="random"):
        return np.array(random.choices([0,1,2,3,4],weights= self.pref_dist,k=self.n_jobs))
    
    def gen_preplan(self):
        self.pre_X = np.zeros((self.n_nurses,self.n_days,self.n_jobs))
        for n in range(self.n_nurses):
            self.pre_X[n] = self.gen_holidayset(n)
        
    def gen_holidayset(self, n):
        plan = np.zeros((self.n_days,self.n_jobs))
        self.nurse_params[n]["holidays"] = bernoulli.rvs(6/52,size = self.n_weeks)
        for i in range(self.nurse_params[n]["holidays"].shape[0]):
            if self.nurse_params[n]["holidays"][i] == 1:
                plan[i*7:(i+1)*7,0] = [1,1,1,1,1,1,1] #full week holiday
        return plan
    
    def gen_rotations(self):
        self.R = np.zeros((self.n_nurses,self.n_days,self.n_jobs))
        for n in range(self.n_nurses):
            self.R[n] = self.gen_rotationset(n)
        
    def gen_rotationset(self, n):
        init_rot = random.randint(1,self.n_standorte-1) #random start standort
        #chance of mid month change = 0.4
        rest_prob = [0.4/(self.n_standorte-2)]*(self.n_standorte-1)
        rest_prob[init_rot-1] = self.prob_rot_whole_window
        self.nurse_params[n]["rotationen"] = [init_rot,random.choices(range(1,self.n_standorte),weights=rest_prob,k=1)[0]]
        rotation = np.zeros((self.n_days,self.n_jobs))
        rotation[:self.n_days//2,self.job_standorte == self.nurse_params[n]["rotationen"][0]] = 1
        rotation[self.n_days//2:,self.job_standorte ==self.nurse_params[n]["rotationen"][1]] = 1
        return rotation
    
    def gen_wishes(self):
        self.W = np.zeros((self.n_nurses,self.n_days,self.n_jobs))
        for n in range(self.n_nurses):
            self.W[n] = self.gen_wishset(n)
    
    def gen_wishset(self, n,type="random"):
        wish_types = random.choices(range(self.n_job_types +1), weights= self.wish_dist, k=self.n_days)
        wishes = np.zeros((self.n_days,self.n_jobs))
        for i in range(self.n_days):
            if wish_types[i] == 0:
                wishes[i] = np.zeros(self.n_jobs)
            else:
                wishes[i][self.n_job_types == wish_types[i]] = 1
        return wishes
    
    def calc_weight(self,nurse: int, day:int, job:int):
        return self.S[job, nurse] * (self.pref_weight*self.prefs[nurse,job] + self.rotation_weight * self.R[nurse,day,job] + self.wish_weight * self.W[nurse,day,job]) 
    
    def nurse_weights(self,nurse) -> np.ndarray:
        weights = np.zeros((self.n_days,self.n_jobs))
        weights += self.pref_weight*self.prefs[nurse]
        weights += self.rotation_weight*self.R[nurse]
        weights += self.wish_weight*self.W[nurse]
        weights = self.S[:,nurse]*weights
        return weights

    def add_model_constraints(self):
        for constraint in self.constraints:
             constraint()
    
    def check_one_job_per_day_per_nurse(self):
        for n in range(self.n_nurses):
            for d in range(self.n_days):
                self.model.addConstr(gp.quicksum(self.X[n,d,j] for j in range(self.n_jobs)) <= 1, "one_job_per_day_per_nurse_"+str(n)+"_"+str(d))
    
    def check_solution_meets_job_demand_per_day(self):
        for j in range(self.n_jobs):
            for d in range(self.n_days):
                self.model.addConstr(gp.quicksum(self.X[n,d,j] for n in range(self.n_nurses)) == self.D[j,d], "job_demand_per_day_"+str(j)+"_"+str(d))
    
    def check_skill_requirement_per_job(self):
        for n in range(self.n_nurses):
            for j in range(self.n_jobs):
                self.model.addConstr(gp.quicksum(self.X[n,d,j] for d in range(self.n_days)) <= self.S[j,n], "skill_requirement_per_job_"+str(n)+"_"+str(j))
    
    def check_pre_plan(self):
        for n in range(self.n_nurses):
            for d in range(self.n_days):
                for j in range(self.n_jobs):
                    self.X[n,d,j].LB = self.pre_X[n,d,j]

    
    def check_shifts(self):
        for job in self.shifts:
            i = 0
            for shift in self.shifts[job]:
                self.check_shift(job,shift,i)
                i += 1


    def check_shift(self,job, shift,i):
        for n in range(self.n_nurses):
            y = self.model.addVar(vtype=gp.GRB.BINARY, name="y_"+str(job)+"_"+str(n)+"_"+str(i)) #control_variable for shift
            self.model.addConstr(gp.quicksum(self.X[n,shift,job]) - y*shift.sum() <= 0, "shift_y_lb_"+str(n)+"_"+str(job)+"_"+str(i)) 
            self.model.addConstr(gp.quicksum(self.X[n,shift,job]) - y*shift.sum() >= 0, "shift_met_"+str(n)+"_"+str(job)+"_"+str(i))
    
    def check_kernels(self):
        for k in range(len(self.hard_kernels)):
            for n in range(self.n_nurses):
                self.check_kernel(n,self.hard_kernels[k]["kernel"],self.hard_kernels[k]["condition"],self.hard_kernels[k]["padding"], self.hard_kernels[k]["name"])
    
    def set_time_before(self,before):
        self.before = before
    
    def set_time_after(self,after):
        self.after = after

    def check_kernel(self,nurse,kernel,condition,padding,name):
        if padding == 0:
            return self.check_kernel_no_padding(nurse,kernel,condition,name)
        else:
            return self.check_kernel_with_padding(nurse,kernel,condition,name,padding)

    def check_kernel_with_padding(self,nurse,kernel,condition,name,padding):
        for d in range(self.n_days):
            self.model.addConstr(self.apply_before_after(kernel, self.X[nurse,:,:],self.before[nurse,:,:],self.after[nurse,:,:],padding,d) <=condition, name + "_"+str(nurse)+"_"+str(d))
    
    def check_kernel_no_padding(self,nurse,kernel,condition,name):
        shape = kernel.shape
        for d in range(self.n_days-shape[0]):
            self.model.addConstr(self.gurobi_convolve(kernel, self.X[nurse,d:d+shape[0],:]) <=condition, name+"_"+str(nurse)+"_"+str(d))

    def gurobi_convolve(self,kernel,x):
        if kernel.shape != x.shape:
            print(kernel.shape,x.shape)
        assert(kernel.shape == x.shape)

        s = 0
        for index, val in np.ndenumerate(kernel):
            s += val * x[index]
        return s

    def check_break_all_ten_days(self):
        self.create_job_sum_check([1],1,0,10)

    def create_job_sum_check(self,jobs:list,min=0,max=0,window=1):
        for nurse in range(self.n_nurses):
            self.check_job_sum_for_one_nurse(nurse,jobs,min,max,window)
    
    def check_job_sum_for_one_nurse(self,nurse,jobs:list,min=0,max=0,window=1):
        if min == max:
            self.model.addConstr(self.X[nurse,:,jobs].sum() == min)
        else: 
            for d in range(self.n_days):
                if d + window >= self.n_days:
                    after = self.after[nurse,:d+window-self.n_days,jobs].sum()
                else:
                    after = 0
                if d - window < 0:
                    before = self.before[nurse,d-window:,jobs].sum()
                else:
                    before = 0
                if max > min:
                    self.model.addConstr(self.X[nurse,:,jobs].sum() <= max - before - after)
                if min >0:
                    self.model.addConstr(self.X[nurse,:,jobs].sum() >= min - before - after)

    
    def check_no_weekends_in_a_row(self):
        for n in range(self.n_nurses):
            self.check_no_weekends_in_a_row_for_one_nurse(n)
                
    
    def check_no_weekends_in_a_row_for_one_nurse(self,nurse):
        for week in range(self.n_weeks):
            if week ==0:
                before = self.before[nurse,[5,6,12,13],2:].sum()
                self.model.addConstr(self.X[nurse,[5,6],2:].sum()< 5 - before)
            if week == 1:
                before = self.before[nurse,[12,13],:].sum()
                self.model.addConstr(self.X[nurse,[5,6,12,13],2:].sum()< 5 - before)
            else:
                weekends = [[5 + 7*(week-i),6+ 7*(week-i)] for i in range(3)].flatten()
                self.model.addConstr(self.X[nurse,weekends,2:].sum()< 5) 
    
    def init_objective(self):
        self.W = np.zeros((self.n_nurses,self.n_days,self.n_jobs))
        for nurse in range(self.n_nurses):
            self.W[nurse] = self.nurse_weights(nurse)
        self.objective = self.gurobi_convolve(self.W,self.X)
        self.init_soft_kernel_scores()
        self.model.setObjective(self.objective, gp.GRB.MAXIMIZE)

    def init_soft_kernel_scores(self):
        self.soft_penalty = self.model.addMVar((self.n_nurses,len(self.soft_kernels),self.n_days),vtype=gp.GRB.BINARY)
        for n in range(self.n_nurses):
            self.add_soft_penalty_for_one_nurse(n)
    
    def add_soft_penalty_for_one_nurse(self,nurse):
        for k in range(len(self.soft_kernels)):
            if self.soft_kernels[k]["padding"] ==0:
                self.soft_penalty_no_padding(nurse,k,self.soft_kernels[k]["kernel"],self.soft_kernels[k]["weight"],self.soft_kernels[k]["offset"],self.soft_kernels[k]["name"])
            else:
                self.soft_penalty_with_padding(nurse,k,self.soft_kernels[k]["kernel"],self.soft_kernels[k]["weight"],self.soft_kernels[k]["padding"],self.soft_kernels[k]["offset"],self.soft_kernels[k]["name"])
                
    def soft_penalty_no_padding(self,nurse,k,kernel,weight,offset,name):
        shape = kernel.shape
        for d in range(self.n_days-shape[0]):
            self.model.addConstr(self.gurobi_convolve(kernel, self.X[nurse,d:d+shape[0],:])  -self.soft_penalty[nurse,k,d]<= offset , name+"_"+str(nurse)+"_"+str(d))
            self.objective += weight * self.soft_penalty[nurse,k,d]
    
    def soft_penalty_with_padding(self,nurse,k,kernel,weight,padding,offset,name):
        for d in range(self.n_days):
            self.model.addConstr(self.apply_before_after(kernel,self.X[nurse,:,:],self.before[nurse,:,:],self.after[nurse,:,:],padding,d) -self.soft_penalty[nurse,k,d]<= offset , name+"_"+str(nurse)+"_"+str(d))
            self.objective += weight * self.soft_penalty[nurse,k,d]
            
    def initialize_start(self):
        self.start = np.zeros((self.n_nurses,self.n_days,self.n_jobs))
        self.job_options = np.sum(self.S,axis=1)
        self.demand = np.copy(self.D)
        

    def apply_before_after(self,k,x,before,after,padding,d):
        s = 0
        if d-padding<0:
            s = (k[:padding-d,:] * before[d-padding:,:]).sum()
            res = self.gurobi_convolve(k[padding-d:,:], x[:d+padding+1,:])
        elif d + padding >= x.shape[0]:
            res = self.gurobi_convolve(k[d-padding-x.shape[0]:,:], x[d-padding:,:])
            s = (k[:d-padding-x.shape[0],:] * after[:d+padding+1-x.shape[0],:]).sum()
        else:
            res = self.gurobi_convolve(k , x[d-padding:d+padding+1,:])
        return res + s
