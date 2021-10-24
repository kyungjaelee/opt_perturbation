import numpy as np
from scipy.special import gamma
from estimation.heavy_tail_observations import WeibullNoise, FrechetNoise, ParetoNoise

class Rewards():
    def __init__(self,means,p,scale,noise_type='pareto',both_side=True):
        self.K = len(means)
        self.means = means
        self.scale = scale
        if noise_type == 'weibull':
            weibull_noise = WeibullNoise(alpha=p, scale=scale,p=p, both_side=both_side)
            self.nu = np.max((weibull_noise.nu_p**(1./p) + np.abs(means - weibull_noise.mean))**p)
            self.noise_generator = lambda : weibull_noise.sample()
        elif noise_type == 'frechet':
            frechet_noise = FrechetNoise(alpha=p+0.1, scale=scale, p=p, both_side=both_side)
            self.nu = np.max((frechet_noise.nu_p**(1./p) + np.abs(means - frechet_noise.mean))**p)
            self.noise_generator = lambda : frechet_noise.sample()
        elif noise_type == 'pareto':
            pareto_noise = ParetoNoise(alpha=p+0.1, scale=scale, p=p, both_side=both_side)
            self.nu = np.max((pareto_noise.nu_p**(1./p) + np.abs(means - pareto_noise.mean))**p)
            self.noise_generator = lambda : pareto_noise.sample()
            
    def get_observations(self):
        return [mean + self.scale*self.noise_generator() for mean in self.means]
