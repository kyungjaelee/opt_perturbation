import numpy as np
from scipy.special import gamma
from estimation.heavy_tail_observations import weibull_noise, frechet_noise, pareto_noise, gaussian_noise

class Rewards():
    def __init__(self,means,p,scale,noise_type='pareto'):
        self.K = len(means)
        self.means = means
        self.scale = scale
        if noise_type == 'weibull':
            self.alpha = p
            self.nu = (scale0 + np.abs(np.max(means)-scale*gamma(1.+1./self.alpha)))**p
            self.noise_generator = lambda: weibull_noise(self.alpha)
        elif noise_type == 'frechet':
            self.alpha = p+0.05
            self.nu = (scale*gamma(1.-p/self.alpha)**(1./p) + np.abs(np.max(means)-scale*gamma(1.-1./self.alpha)))**p
            self.noise_generator = lambda: frechet_noise(self.alpha)
        elif noise_type == 'pareto':
            self.alpha = p+0.05
            self.nu = (scale**(self.alpha/p)*(self.alpha/(self.alpha-p))**(1./p) + np.abs(np.max(means)-scale*self.alpha/(self.alpha-1.)))**p
            self.noise_generator = lambda: pareto_noise(self.alpha)
    def get_observations(self):
        return [mean + self.scale*self.noise_generator() for mean in self.means]
