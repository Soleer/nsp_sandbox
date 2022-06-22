from tomlkit import integer, string
import numpy as np
import math
from scipy.stats import bernoulli


class NSPbuilder:
    print_info: bool
    version: str
    n_nurses: int
    n_jobs: int
    n_days: int

    skill_exp_exp=0.35
    skill_exp_std=0.35
    skill_cov_exp=0.017
    skill_cov_std=0.03


    def __init__(self, print_info=True) -> None:
        self.print_info = print_info
        self.version = "0.0.1"
    
    def set_n_nurses(self, n:int) -> None:
        self.n_nurses = n
    
    def set_n_jobs(self, n:int) -> None:
        self.n_jobs = n
        
    def set_n_days(self, n:int) -> None:
        self.n_days = n

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
        times = math.ceil(self.n_days /7.)
        demand = np.tile(week, times )[0:self.n_days]
        return demand
