import numpy as np
from scipy.stats import invweibull
from scipy.special import gamma

class NoiseModel():
    def __init__(self,alpha=1.1, scale=1.):
        self.alpha=1.1
        self.scale=1.
    
class ParetoNoise(NoiseModel):
    def __init__(self,alpha=1.6, scale=1.,p=1.5, both_side=False):
        NoiseModel.__init__(self,alpha,scale)
        self.p = p
        self.mean = scale*alpha/(alpha-1.)
        self.nu_p = (scale**p)*alpha/(alpha-p)
        self.both_side = both_side
        
    def sample(self):
        if not self.both_side or np.random.uniform() > 0.5:
            return self.scale*np.random.pareto(self.alpha) - self.mean
        else:
            return -self.scale*np.random.pareto(self.alpha) + self.mean

class WeibullNoise(NoiseModel):
    def __init__(self,alpha=1.6, scale=1.,p=1.5, both_side=False):
        NoiseModel.__init__(self,alpha,scale)
        self.p = p
        self.mean = scale*gamma(1.+1./self.alpha)
        self.nu_p = (scale**p)*gamma(1.+p/alpha)
        self.both_side = both_side

    def sample(self):
        if not self.both_side or np.random.uniform() > 0.5:
            return self.scale*np.random.weibull(self.alpha) - self.mean
        else:
            return -self.scale*np.random.weibull(self.alpha) + self.mean
        
class FrechetNoise(NoiseModel):
    def __init__(self,alpha=1.6, scale=1.,p=1.5, both_side=False):
        NoiseModel.__init__(self,alpha,scale)
        self.p = p
        self.mean = scale*gamma(1.-1./alpha)
        self.nu_p = (scale**p)*gamma(1.-p/alpha)
        self.both_side = both_side

    def sample(self):
        if not self.both_side or np.random.uniform() > 0.5:
            return self.scale*invweibull.rvs(self.alpha,scale=1.) - self.mean
        else:
            return -self.scale*invweibull.rvs(self.alpha,scale=1.) + self.mean