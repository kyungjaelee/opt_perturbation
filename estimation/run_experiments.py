import argparse
import numpy as np
from scipy.special import gamma

from estimation.robust_estimator import SampleMean, TruncatedMean, MedianofMean, CatoniMean, WeaklyRobustMean
from estimation.heavy_tail_observations import weibull_observation, frechet_observation, pareto_observation, gaussian_observation

parser = argparse.ArgumentParser(description="Run experiments on various robust mean estimators under heavy tailed noise")
parser.add_argument('--noise', metavar='dist', type=str, default='pareto', choices=['weibull', 'frechet', 'pareto', 'gaussian'], help='A type of noise')
parser.add_argument('--moment', metavar='p', type=float, default=1.5, help='A maximum bounded moment')
parser.add_argument('--scale', metavar='s', type=float, default=1.0, help='Scale of noise')
parser.add_argument('--mean', metavar='m', type=float, default=1.0, help='True mean')
parser.add_argument('--seed', metavar='i', type=int, default=0, help='A random seed')
parser.add_argument('--samples', metavar='n', type=int, default=10000, help='Samples')
parser.add_argument('--steps', metavar='t', type=int, default=1000, help='Steps')

args = parser.parse_args()

seed = args.seed
samples = args.samples
steps = args.steps
noise_type = args.noise
p = args.moment
scale = args.scale
mean = args.mean

if noise_type == 'weibull':
    alpha = p
    nu = scale**p  
    get_observation = lambda : weibull_observation(mean,scale,alpha)
elif noise_type == 'frechet':
    alpha = p+0.05
    nu = (scale*gamma(1-p/alpha)**(1./p) + np.abs(mean - gamma(1.-1./alpha)))**p
    get_observation = lambda : frechet_observation(mean,scale,alpha)
elif noise_type == 'pareto':
    alpha = p+0.05
    nu = (scale**(alpha/p)*(alpha/(alpha-p))**(1./p) + np.abs(mean - gamma(1.+1./alpha)))**p
    get_observation = lambda : pareto_observation(mean,scale,alpha)
elif noise_type == 'gaussian':
    nu = scale**2
    get_observation = lambda : gaussian_observation(mean,scale)
    
sample_mean = SampleMean(nu, p)
trunc_mean = TruncatedMean(nu, p, delta=1.0, schedule=True)
median_mean = MedianofMean(nu,p, delta=1.0, schedule=True)
catoni_mean = CatoniMean(nu,p, delta=1.0, schedule=True)
weakly_robust_mean = WeaklyRobustMean(nu,p)

# trunc_mean = TruncatedMean(nu, p)
# median_mean = MedianofMean(nu,p)
# catoni_mean = CatoniMean(nu,p)

sample_mean_error_list = []
trunc_mean_error_list = []
median_mean_error_list = []
catoni_mean_error_list = []
weakly_robust_mean_error_list = []

np.random.seed(seed)
for i in range(samples):
    y = get_observation()

    sample_mean.update(y)
    trunc_mean.update(y)
    median_mean.update(y)
    catoni_mean.update(y)
    weakly_robust_mean.update(y)

    if ((i+1)%steps)==0 or i==0:
        y_hat = sample_mean.predict()
        sample_mean_error_list.append(np.abs(mean-y_hat))
        y_hat = trunc_mean.predict()
        trunc_mean_error_list.append(np.abs(mean-y_hat))
        y_hat = median_mean.predict()
        median_mean_error_list.append(np.abs(mean-y_hat))
        y_hat = catoni_mean.predict()
        catoni_mean_error_list.append(np.abs(mean-y_hat))
        y_hat = weakly_robust_mean.predict()
        weakly_robust_mean_error_list.append(np.abs(mean-y_hat))
        
print("Noise - {}, Moment - {:.2f}, Nu = {:.2f}".format(noise_type, p, nu))
print("Sample Mean Error : {:.3f}".format(sample_mean_error_list[-1]))
print("Truncated Mean Error : {:.3f}".format(trunc_mean_error_list[-1]))
print("Median of Mean Error : {:.3f}".format(median_mean_error_list[-1]))
print("Catoni's Mean Error : {:.3f}".format(catoni_mean_error_list[-1]))
print("Weakly Robust Mean Error : {:.3f}".format(weakly_robust_mean_error_list[-1]))
        
filename = 'estimation_results/{:}-p{:.2f}-s{:.2f}-m{:.2f}-size{:d}-seed{:d}.npy'.format(noise_type,p,scale,mean,samples,seed)
with open(filename,'wb') as f:
    np.savez(f, 
             nu=nu,
             alpha=alpha,
             sample_mean=sample_mean_error_list,
             trunc_mean=trunc_mean_error_list,
             median_mean=median_mean_error_list,
             catoni_mean=catoni_mean_error_list,
             weakly_robust_mean=weakly_robust_mean_error_list
            )
    print('Data saved at {}'.format(filename))
