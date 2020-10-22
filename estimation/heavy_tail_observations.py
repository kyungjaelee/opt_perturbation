import numpy as np
from scipy.stats import invweibull
from scipy.special import gamma

# Generate noisy observations
def weibull_observation(mean, scale, alpha):        
    return weibull_noise(alpha)*scale + mean

def frechet_observation(mean, scale, alpha):        
    return frechet_noise(alpha)*scale + mean

def both_side_frechet_observation(mean, scale, alpha):        
    return both_side_frechet_noise(alpha)*scale + mean

def pareto_observation(mean, scale, alpha):
    return pareto_noise(alpha)*scale + mean

def both_side_pareto_observation(mean, scale, alpha):
    return both_side_pareto_noise(alpha)*scale + mean

def gaussian_observation(mean, scale):
    return gaussian_noise()*scale + mean

# mean zero Noise functions by shifting an original R.V.
def weibull_noise(alpha):        
    return (np.random.weibull(alpha) - gamma(1+1/alpha))

def frechet_noise(alpha):        
    return (invweibull.rvs(alpha,scale=scale) - gamma(1-1/alpha))

def pareto_noise(alpha):
    return (np.random.pareto(alpha) - 1./(alpha-1))

def gaussian_noise():
    return np.random.normal()

# Extend one sided noise to two sided noise by flipping its sign with probability 1/2
def both_side_frechet_noise(alpha):        
    side_variable = np.random.choice(2)
    if side_variable==0:
        return invweibull.rvs(alpha,scale=scale)
    else:
        return -invweibull.rvs(alpha,scale=scale)
    
def both_side_pareto_noise(alpha):
    side_variable = np.random.choice(2)
    if side_variable==0:
        return np.random.pareto(alpha) + 1
    else:
        return -np.random.pareto(alpha) - 1