import numpy as np
import scipy.optimize

def find_truncated_samples(samples, trunc_para):
    """
    Parameters:
    1. samples: np.array(., dtype=float)
        samples of the distribution
    2. trunc_para: float
        truncation parameter
    Returns:
    1. trunc_samples: np.array(., dtype=float)
        sample*Indicator(|sample|<=trunc_para)
    """
    trunc_samples = np.copy(samples)
    abs_samples = np.abs(trunc_samples)
    trunc_samples[np.where(abs_samples>trunc_para)] = 0
    return trunc_samples

def find_truncated_mean(samples, trunc_para):
    """
    Parameters:
    1. samples: np.array(., dtype=float)
        samples of the distribution
    2. trunc_para: float
        truncation parameter
    Returns:
    1. : float
        truncated empirical mean (TEM)
    """
    trunc_samples = find_truncated_samples(samples, trunc_para)
    return np.mean(trunc_samples)

# Parent Robust Estimator Class
class RobustEstimator():
    def __init__(self,nu,p):
        self._nu = nu
        self._p = p
        self._y_list = []
        self.y_hat = 0
        self.n = 0
    def predict(self):
        NotImplemented
    def update(self,y):        
        NotImplemented
        
# Sample Mean estiamtor
class SampleMean(RobustEstimator):
    def __init__(self,nu,p):
        RobustEstimator.__init__(self,nu,p)
    def predict(self):
        return self.y_hat
    def update(self,y):
        self._y_list.append(y)
        n = len(self._y_list)
        self.y_hat = (1.-1./n)*self.y_hat + 1./n*y
        self.n = n
    
# Truncated Mean estimator
class TruncatedMean(RobustEstimator):
    def __init__(self,nu,p, delta=1e-4, schedule=False):
        RobustEstimator.__init__(self,nu,p)
        self._delta = delta
        self._schedule = schedule
    def predict(self):
        return self.y_hat
    def update(self,y):
        self._y_list.append(y)
        n = len(self._y_list)
        if self._schedule:
            self._delta = np.maximum(1.0/(n**2),1e-4)
        
        self.y_hat=np.sum([y for y in self._y_list if np.abs(y) <= (self._nu*n/2./np.maximum(np.log(1./self._delta),1.0))**(1/self._p)])/n
        self.n = n
        
    def update_delta(self,new_delta):
        self._delta=new_delta
                
# Median of Mean estimator
class MedianofMean(RobustEstimator):
    def __init__(self,nu,p, delta=1e-4, schedule=False):
        RobustEstimator.__init__(self,nu,p)
        self._delta = delta
        self._schedule = schedule
    def predict(self):
        return self.y_hat
    def update(self,y):
        self._y_list.append(y)
        n = len(self._y_list)
        if self._schedule:
            self._delta = np.maximum(1.0/(n**2),1e-4)
        
        k = np.maximum(int(np.minimum(8*np.log(np.exp(1/8.)/self._delta),n/2.)),1)
        N = int(n/k)
        mean_list = np.zeros(k)
        for i in range(k):
            mean_list[i] = np.mean(self._y_list[i*N:(i+1)*N])
        self.y_hat = np.median(mean_list)
        self.k = k
        self.N = N        
        self.n = n
    def update_delta(self,new_delta):
        self._delta=new_delta

# Static helper functions for Catoni's estimator
def _cal_ap(p):
    if p == 2:
        return 1./2.
    else:
        ap = (2.*(((2.-p)/(p-1.)) ** (1.-(2./p))) + ((2.-p)/(p-1))**(2.-(2./p))) **(-p/2.)
        return ap

def _cal_psi(x, p):
    ap = _cal_ap(p)
    psi = np.sign(x)*np.log(ap* (abs(x)**p) + np.sign(x)*x + 1.)
    return psi
        
def _cal_alpha(p,nu,mean_bnd,delta,n):
    ap = _cal_ap(p)
    if n > 2.**(p/(p-1.))*ap**(1./(p-1.))*np.log(1./delta):
        alpha = (np.log(1./delta)/ap/n)**(1/p)*(1.-2.*ap**(1./p)*(np.log(1./delta)/n)**((p-1.)/p))/(nu**(1./p)+mean_bnd)
    else:
        alpha = 1.0
    return alpha

# Catoni's M estimator
class CatoniMean(RobustEstimator):
    def __init__(self,nu,p,delta=1e-4,mean_bnd=1.0,schedule=False):
        RobustEstimator.__init__(self,nu,p)
        self._delta = delta
        self._mean_bnd = mean_bnd
        self._schedule = schedule

    def predict(self):
        return self.y_hat
    
    def update(self,y):
        self._y_list.append(y)
        n = np.maximum(len(self._y_list),1.)
        if self._schedule:
            self._delta = np.maximum(1.0/(n**2),1e-4)
            
        alpha = _cal_alpha(self._p,self._nu,self._mean_bnd,self._delta,n)
        remainder = lambda x: np.sum(_cal_psi(alpha*(np.array(self._y_list) - x),self._p))/alpha/n
        sol = scipy.optimize.root_scalar(remainder,x0=0.,x1=10.0)
        self.y_hat = sol.root        
        self.n = n
    def update_delta(self,new_delta):
        self._delta=new_delta
        
# Weakly Robust Mean estimator
class WeaklyRobustMean(RobustEstimator):
    def __init__(self,nu,p,c=1.0):
        RobustEstimator.__init__(self,nu,p)
        self._c = c
        
    def predict(self):
        return self.y_hat
    
    def update(self,y):
        self._y_list.append(y)
        n = np.maximum(len(self._y_list),1.)
        self.y_hat = np.sum(_cal_psi(np.array(self._y_list)/self._c/(n**(1./self._p)),self._p))*self._c/(n**(1.-1./self._p))
        self.n = n
