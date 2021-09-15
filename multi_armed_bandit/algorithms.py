import numpy as np
from scipy.stats import invweibull
from scipy.special import gamma

from estimation.robust_estimator import SampleMean, TruncatedMean, MedianofMean, CatoniMean, WeaklyRobustMean, _cal_ap, find_truncated_mean
from utils.custom_distribution import random_GEV

class MAB_model:
    def __init__(self, T, K, p, nu, estimator_type='TruncatedMean', name='MAB_algorithm'):
        # Number of arms
        self.T = T
        self.K = K
        self.p = p
        self.nu = nu
        self.name = name

        self.estimator_type = estimator_type        
        if estimator_type == 'SampleMean':
            estimators = [SampleMean(nu, p) for _ in range(K)]
        elif estimator_type == 'TruncatedMean':
            estimators = [TruncatedMean(nu, p, delta=1.0, schedule=True) for _ in range(K)]
        elif estimator_type == 'MedianofMean':
            estimators = [MedianofMean(nu, p, delta=1.0, schedule=True) for _ in range(K)]
        elif estimator_type == 'CatoniMean':
            estimators = [CatoniMean(nu, p, delta=1.0, schedule=True) for _ in range(K)]
        elif estimator_type == 'WeaklyRobustMean':
            estimators = [WeaklyRobustMean(nu, p, c=0.1) for _ in range(K)]
        self.reward_estimators = estimators
        
    def choose(self, step):
        raise NotImplementedError
        
    def update(self, a, r, step):
        raise NotImplementedError

class RobustUCB(MAB_model):
    def __init__(self, T, K, p, nu, c=1.0, estimator_type='TruncatedMean'):
        MAB_model.__init__(self, T, K, p, nu, estimator_type, 'Robust UCB')
        
        self.c = c
        if self.estimator_type == 'TruncatedMean':
            self.eta = 4.**(self.p/(self.p-1.))
            self.u = nu
            self.b0 = 0.0
        elif self.estimator_type == 'MedianofMean':
            self.eta = 16.0
            self.u = 12.0*nu
            self.b0 = 2.0
        elif self.estimator_type == 'CatoniMean':
            ap = _cal_ap(self.p)
            self.eta = 4.**(self.p/(self.p-1.))
            self.u = ap*nu
            self.b0 = 0.0
        
    def choose(self, step):
        if step < self.K:
            a = step
        else:
            confidences = [reward_estimator.predict() +
                           self.c*(self.u**(1/self.p)) * ((self.eta*np.log(step**2) + self.b0)/ reward_estimator.n)**(1.-1./self.p) for reward_estimator in self.reward_estimators]
            a = np.argmax(confidences)
        return a
    
    def update(self, a, r, step):
        self.reward_estimators[a].update(r)
        if self.estimator_type in ['TruncatedMean','MedianofMean','CatoniMean']:
            for estimator in self.reward_estimators:
                estimator.update_delta(np.maximum((step+1.)**(-2.),1e-4))

class DSEE(MAB_model):
    def __init__(self, T, K, p, nu, c=1.0, estimator_type='TruncatedMean'):
        MAB_model.__init__(self, T, K, p, nu, estimator_type, 'DSEE')
        self.exploration_n = 0
        
        self.c = c
        self.w = 1./(4**(p/(p-1.))*nu**(1/(p-1.))*c**(p/(p-1.)))

    def choose(self, step):
        if self.exploration_n < self.K * self.w * np.log(step+1.):
            a = (self.exploration_n % self.K)
            self.exploration_n += 1
        else:
            a = np.argmax([reward_estimator.predict() for reward_estimator in self.reward_estimators])
        return a

    def update(self, a, r, step):
        self.reward_estimators[a].update(r)
        if self.estimator_type in ['TruncatedMean','MedianofMean','CatoniMean']:
            for estimator in self.reward_estimators:
                estimator.update_delta(np.maximum(np.exp(-estimator.n/self.w),1e-6))

class GSR():
    def __init__(self, T, K, p, nu, q=0.5):
        self.nu = nu
        self.p = p
        self.K = K
        norm = 0.5 + np.sum([1/i for i in range(2,K+1)])
        self.n_arr = np.array([np.floor((T-K)/((K+1-k)*norm)) for k in range(1,K)], dtype=int) 
        self.remaining_set = np.arange(K)
        self.sampled_rewards = np.zeros((K, self.n_arr[-1]))
        self.action_cnt = np.zeros(K)
        self.curr_phase = 0
        self.curr_idx = 0
        self.q = q
        self.name = 'DSEE'
    def choose(self, step):
        k = self.curr_phase
        if k < self.K-1:
            a = self.remaining_set[self.curr_idx]
            if self.action_cnt[a] == self.n_arr[k]:
                if self.curr_idx == len(self.remaining_set)-1:
                    self.curr_idx = 0
                    self.curr_phase = k+1
                    
                    trunc_para = self.n_arr[k]**self.q
                    self.reward_estimators = np.array([find_truncated_mean(self.sampled_rewards[i,:self.n_arr[k]],
                        trunc_para) for i in self.remaining_set])
                    removed_ind = np.argmin(self.reward_estimators)
                    self.remaining_set = np.append(self.remaining_set[:removed_ind], self.remaining_set[removed_ind+1:])   
                else:
                    self.curr_idx += 1

                a = self.remaining_set[self.curr_idx]
        else:
            a = self.remaining_set[0]
        return a
    
    def update(self, a, r, step):
        if self.curr_phase < self.K-1:
            self.sampled_rewards[a,int(self.action_cnt[a])] = r
            self.action_cnt[a] += 1
            
class ModifiedRobustUCB(MAB_model):
    def __init__(self, T, K, p, nu, c=1.0, estimator_type='WeaklyRobustMean'):
        MAB_model.__init__(self, T, K, p, nu, estimator_type, 'MR-UCB')        
        self.c = c
        
    def choose(self, step):
        if step < self.K:
            a = step
        else:
            confidences = [reward_estimator.predict() +
                           self.c*np.log(np.maximum(self.T/self.K/reward_estimator.n,1.))/ reward_estimator.n**(1.-1./self.p) for reward_estimator in self.reward_estimators]
            a = np.argmax(confidences)
        return a
    
    def update(self, a, r, step):
        self.reward_estimators[a].update(r)
        if self.estimator_type in ['TruncatedMean','MedianofMean','CatoniMean']:
            for estimator in self.reward_estimators:
                estimator.update_delta(np.maximum((step+1.)**(-2.),1e-4))
            
class APE(MAB_model):
    def __init__(self, T, K, p, nu, c=1.0, c_est=100000., estimator_type='WeaklyRobustMean', perturbation={'perturbation_type':'Pareto','params':{'alpha':4.,'scale':1.}}):
        MAB_model.__init__(self, T, K, p, nu, estimator_type, 'APE-'+perturbation['perturbation_type'])
        self.c=c
        for reward_estimator in self.reward_estimators:
            reward_estimator._c = c_est
        
        self.perturbation = perturbation
        if perturbation['perturbation_type'] == 'Weibull':
            alpha = perturbation['params']['k']
            scale = perturbation['params']['scale']
            self.perturbations = [lambda : np.random.weibull(alpha)*scale for _ in range(K)]
        elif perturbation['perturbation_type'] == 'Frechet':
            alpha = perturbation['params']['alpha']
            scale = perturbation['params']['scale']
            self.perturbations = [lambda : invweibull.rvs(alpha,scale=scale) for _ in range(K)]
        elif perturbation['perturbation_type'] == 'Pareto':
            alpha = perturbation['params']['alpha']
            scale = perturbation['params']['scale']
            self.perturbations = [lambda : (np.random.pareto(alpha)+1.)*scale for _ in range(K)]
        elif perturbation['perturbation_type'] == 'Gamma':
            alpha = perturbation['params']['alpha']
            scale = perturbation['params']['scale']
            self.perturbations = [lambda : np.random.gamma(alpha)*scale for _ in range(K)]
        elif perturbation['perturbation_type'] == 'GEV':
            zeta = perturbation['params']['zeta']
            scale = perturbation['params']['scale']
            self.perturbations = [lambda : random_GEV(zeta)*scale for _ in range(K)]
        elif perturbation['perturbation_type'] == 'Bounded':
            self.perturbations = [lambda : 2.*np.random.uniform() - 1. for _ in range(K)]
           
    def choose(self, step):
        if step < self.K:
            a = step
        else:
            if self.perturbation['perturbation_type'] == 'Bounded':
                confidences = [reward_estimator.predict() + self.c*perturbation()*np.log(np.maximum(self.T/self.K/reward_estimator.n,1.))/(reward_estimator.n**(1.-1./self.p)) for reward_estimator,perturbation in zip(self.reward_estimators,self.perturbations)]
            else:
                confidences = [reward_estimator.predict() + self.c*perturbation()/(reward_estimator.n**(1.-1./self.p)) for reward_estimator,perturbation in zip(self.reward_estimators,self.perturbations)]
                
            a = np.argmax(confidences)
        return a
    
    def update(self, a, r, step):
        self.reward_estimators[a].update(r)