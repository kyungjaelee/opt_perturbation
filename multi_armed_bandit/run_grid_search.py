import argparse
import numpy as np

from multi_armed_bandit.reward_functions import Rewards
from multi_armed_bandit.algorithms import RobustUCB, APE, DSEE, GSR, ModifiedRobustUCB

parser = argparse.ArgumentParser(description="Run experiments on various robust mean estimators under heavy tailed noise")
parser.add_argument('--noise', metavar='dist', type=str, default='pareto', choices=['weibull', 'frechet', 'pareto', 'gaussian'], help='A type of noise')
parser.add_argument('--moment_param', metavar='q', type=float, default=1.5, help='An user defined maximum bounded moment')
parser.add_argument('--moment', metavar='p', type=float, default=1.5, help='A maximum bounded moment')
parser.add_argument('--scale', metavar='s', type=float, default=1.0, help='Scale of noise')
parser.add_argument('--mean', metavar='m', type=str, default='one_hot', choices=['one_hot', 'random', 'sequence'], help='True mean')
parser.add_argument('--gap', metavar='g', type=float, default=0.1, help='Gap')
parser.add_argument('--seed', metavar='i', type=int, default=1, help='A random seed')
parser.add_argument('--samples', metavar='n', type=int, default=50000, help='Samples')
parser.add_argument('--arms', metavar='K', type=int, default=100, help='The munber of arms')
parser.add_argument('--algo', metavar='alg', type=str, default='ape-pareto', help='Algorithm')

args = parser.parse_args()

seed = args.seed
samples = args.samples

mean_type = args.mean
noise_type = args.noise
q = args.moment_param
p = args.moment
scale = args.scale
K = args.arms

# How to define mean of rewards
if mean_type == 'one_hot':
    gap= args.gap
    means = (1.-gap)*np.ones(K)
    opt_a = np.random.choice(K)
    means[opt_a]+=gap
elif mean_type == 'random':
    means = np.random.uniform(0., 1., K)
elif mean_type == 'sequence':
    means = np.linspace(0., 1., K)
    
rewards_generator = Rewards(means,p,scale,noise_type)
nu = rewards_generator.nu

algos = []
algos_name = []
algos_type = args.algo
if algos_type == 'ape-weibull':
    c_list = 10**np.linspace(-3.,2.,50)
    for i in range(len(c_list)):
        algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Weibull','params':{'k':1.0,'scale':1.0}}))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},k:{:.2f},$\\lambda$:{:.2f})".format(algos_type,q,c_list[i],1.0,1.0))
elif algos_type == 'ape-frechet':
    c_list = 10**np.linspace(-3.,2.,50)
    for i in range(len(c_list)):
        if p**2./(p-1.) > np.log(K):
            algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Frechet','params':{'alpha':p**2./(p-1.),'scale':1.0}}))
            algos_name.append("{:}(q:{:.1f},c:{:.4f},$\\alpha$:{:.2f},$\\lambda$:{:.2f})".format(algos_type,q,c_list[i],p**2./(p-1.),1.0))
        else:
            algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Frechet','params':{'alpha':np.log(K),'scale':np.log(K)}}))
            algos_name.append("{:}(q:{:.1f},c:{:.4f},$\\alpha$:{:.2f},$\\lambda$:{:.2f})".format(algos_type,q,c_list[i],np.log(K),np.log(K)))
elif algos_type == 'ape-pareto':    
    c_list = 10**np.linspace(-3.,2.,50)
    for i in range(len(c_list)):
        if p**2./(p-1.) > np.log(K):
            algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Pareto','params':{'alpha':p**2./(p-1.),'scale':1.0}}))
            algos_name.append("{:}(q:{:.1f},c:{:.4f},$\\alpha$:{:.2f},$\\lambda$:{:.2f})".format(algos_type,q,c_list[i],p**2./(p-1.),1.0))
        else:
            algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Pareto','params':{'alpha':np.log(K),'scale':np.log(K)}}))
            algos_name.append("{:}(q:{:.1f},c:{:.4f},$\\alpha$:{:.2f},$\\lambda$:{:.2f})".format(algos_type,q,c_list[i],np.log(K),np.log(K)))
elif algos_type == 'ape-gamma':
    c_list = 10**np.linspace(-3.,2.,50)
    for i in range(len(c_list)):
        algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Gamma','params':{'alpha':1.0,'scale':1.0}}))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},$\\alpha$:{:.2f},$\\lambda$:{:.2f})".format(algos_type,q,c_list[i],1.0,1.0))
elif algos_type == 'ape-GEV':    
    c_list = 10**np.linspace(-3.,2.,50)
    for i in range(len(c_list)):
        algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'GEV','params':{'zeta':0.0,'scale':1.0}}))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},$\\zeta$:{:.2f},$\\lambda$:{:.2f})".format(algos_type,q,c_list[i],0.0,1.0))
elif algos_type == 'ape-uniform':  
    c_list = 10**np.linspace(-3.,2.,50)             
    for i in range(len(c_list)):
        algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Uniform','params':{}}))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'ape-rademacher':  
    c_list = 10**np.linspace(-3.,2.,50)             
    for i in range(len(c_list)):
        algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Rademacher','params':{}}))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'ape-rademacher2':  
    c_list = 10**np.linspace(-3.,2.,50)             
    for i in range(len(c_list)):
        algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Rademacher2','params':{}}))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'ape-rademacher3':  
    c_list = 10**np.linspace(-3.,2.,50)             
    for i in range(len(c_list)):
        algos.append(APE(samples, K, q, nu, c=c_list[i], perturbation={'perturbation_type':'Rademacher3','params':{}}))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'ucb-truncated-mean':  
    c_list = 10**np.linspace(-3.,2.,50)          
    for i in range(len(c_list)):
        algos.append(RobustUCB(samples, K, q, nu, c=c_list[i], estimator_type='TruncatedMean'))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'ucb-catoni-mean': 
    c_list = 10**np.linspace(-3.,2.,50)             
    for i in range(len(c_list)):
        algos.append(RobustUCB(samples, K, q, nu, c=c_list[i], estimator_type='CatoniMean'))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'ucb-median-of-mean':  
    c_list = 10**np.linspace(-3.,2.,50)         
    for i in range(len(c_list)):
        algos.append(RobustUCB(samples, K, q, nu, c=c_list[i], estimator_type='MedianofMean'))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'mr-ucb':           
    c_list = 10**np.linspace(-3.,2.,50)             
    for i in range(len(c_list)):
        algos.append(ModifiedRobustUCB(samples, K, q, nu, c=c_list[i]))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'dsee': 
    c_list = 10**np.linspace(-3.,2.,50)                 
    for i in range(len(c_list)):
        algos.append(DSEE(samples, K, q, nu, c=c_list[i], estimator_type='TruncatedMean'))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
elif algos_type == 'gsr':
    c_list = 10**np.linspace(-3.,2.,50)                 
    for i in range(len(c_list)):
        algos.append(GSR(samples, K, q, nu, T=samples, q=c_list[i]))
        algos_name.append("{:}(q:{:.1f},c:{:.4f},u:{:.2f})".format(algos_type,q,c_list[i],nu))
                             
total_regret_list = [[] for _ in range(len(algos))]
agv_regret_list = [[] for _ in range(len(algos))]
action_cnt = [np.zeros(K) for _ in range(len(algos))]

np.random.seed(seed)
rewards = np.zeros((samples,K))
for step in range(samples):
    rewards[step] = rewards_generator.get_observations()

for step in range(samples):
    for alg_idx, algo in enumerate(algos):
        a = algo.choose(step)
        r = rewards[step,a]
        algo.update(a,r,step)
        
        regret = np.max(means)- means[a]
        total_regret_list[alg_idx].append(regret)
        agv_regret_list[alg_idx].append(np.mean(total_regret_list[alg_idx]))
        action_cnt[alg_idx][a]+=1
        
if mean_type == 'one_hot':
    filename = 'multi_armed_bandit_results/grid_search_{:}-{:}-{:}-p{:.2f}-s{:.2f}-g{:.2f}-K{:d}-size{:d}-seed{:d}.npy'.format(mean_type,noise_type,algos_type,p,scale,gap,K,samples,seed)
    
with open(filename,'wb') as f:
    np.savez(f, 
             noise_type=noise_type,
             means=means,
             K=K,
             p=p,
             scale=scale,
             nu=nu,
             algos_name=algos_name,
             total_regret_list=total_regret_list,
             agv_regret_list=agv_regret_list,
             action_cnt=action_cnt
            )
    print('Data saved at {}'.format(filename))