import argparse
import numpy as np
from scipy.special import gamma

from estimation.robust_estimator import SampleMean, TruncatedMean, MedianofMean, CatoniMean, WeaklyRobustMean
from estimation.heavy_tail_observations import weibull_observation, frechet_observation, pareto_observation, gaussian_observation

parser = argparse.ArgumentParser(description="Run experiments on various robust mean estimators under heavy tailed noise")
parser.add_argument('--noise', metavar='dist', type=str, default='weibull', choices=['weibull', 'frechet', 'pareto'], help='A type of noise')
parser.add_argument('--side', metavar='side', type=bool, default=False, help='Positive noise? or Real valued noise including negative part?')
parser.add_argument('--moment', metavar='p', type=float, default=1.9, help='A maximum bounded moment')
parser.add_argument('--scale', metavar='s', type=float, default=1.0, help='Scale of noise')
parser.add_argument('--mean', metavar='m', type=float, default=1.0, help='True mean')
parser.add_argument('--seed', metavar='i', type=int, default=0, help='A random seed')
parser.add_argument('--samples', metavar='n', type=int, default=1000, help='Samples')
parser.add_argument('--steps', metavar='t', type=int, default=100, help='Steps')

args = parser.parse_args("")
seed = args.seed
samples = args.samples
steps = args.steps
noise_type = args.noise
both_side = args.side
p = args.moment
scale = args.scale
mean = args.mean

if noise_type == 'weibull':
    weibull_noise = WeibullNoise(alpha=p, scale=scale,p=p, both_side=both_side)
    nu = (weibull_noise.nu_p**(1./p) + np.abs(mean - weibull_noise.mean))**p
    get_observation = lambda : mean + weibull_noise.sample()
elif noise_type == 'frechet':
    frechet_noise = FrechetNoise(alpha=p+0.05, scale=scale, p=p, both_side=both_side)
    nu = (frechet_noise.nu_p**(1./p) + np.abs(mean - frechet_noise.mean))**p
    get_observation = lambda : mean + frechet_noise.sample()
elif noise_type == 'pareto':
    pareto_noise = ParetoNoise(alpha=p+0.05, scale=scale, p=p, both_side=both_side)
    nu = (pareto_noise.nu_p**(1./p) + np.abs(mean - pareto_noise.mean))**p
    get_observation = lambda : mean + pareto_noise.sample()
    
c_list = [1000000., 10000., 100., 1., 0.01, 0.00001]
    
sample_mean = SampleMean(nu, p)
trunc_mean = TruncatedMean(nu, p, delta=1., schedule=True)
median_mean = MedianofMean(nu,p, delta=1., schedule=True)
catoni_mean = CatoniMean(nu,p, delta=1., schedule=True)
weakly_robust_means = [WeaklyRobustMean(nu, p, c=c) for c in c_list]

sample_mean_error_list = []
trunc_mean_error_list = []
median_mean_error_list = []
catoni_mean_error_list = []
weakly_robust_means_error_list = [[] for _ in c_list]

np.random.seed(seed)
for i in range(samples):
    y = get_observation()

    sample_mean.update(y)
    trunc_mean.update(y)
    median_mean.update(y)
    catoni_mean.update(y)
    for weakly_robust_mean in weakly_robust_means:
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
        for weakly_robust_mean, weakly_robust_mean_error_list in zip(weakly_robust_means,weakly_robust_means_error_list):
            y_hat = weakly_robust_mean.predict()
            weakly_robust_mean_error_list.append(np.abs(mean-y_hat))
        
print("Noise - {}, Moment - {:.2f}, Nu = {:.2f}".format(noise_type, p, nu))
print("Sample Mean Error : {:.3f}".format(sample_mean_error_list[-1]))
print("Truncated Mean Error : {:.3f}".format(trunc_mean_error_list[-1]))
print("Median of Mean Error : {:.3f}".format(median_mean_error_list[-1]))
print("Catoni's Mean Error : {:.3f}".format(catoni_mean_error_list[-1]))
for c, weakly_robust_mean_error_list in zip(c_list,weakly_robust_means_error_list):
    print("Weakly Robust Mean Error ({:.3f}) : {:.3f}".format(c, weakly_robust_mean_error_list[-1]))    

filename = 'estimation_results/{:}-p{:.2f}-s{:.2f}-m{:.2f}-size{:d}-seed{:d}.npy'.format(noise_type,p,scale,mean,samples,seed)
with open(filename,'wb') as f:
    np.savez(f, 
             nu=nu,
             alpha=alpha,
             sample_mean=sample_mean_error_list,
             trunc_mean=trunc_mean_error_list,
             median_mean=median_mean_error_list,
             catoni_mean=catoni_mean_error_list,
             weakly_robust_means=weakly_robust_means_error_list
            )
    print('Data saved at {}'.format(filename))