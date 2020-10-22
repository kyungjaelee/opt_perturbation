import numpy as np

def GEV_cdf(x,zeta):
    if zeta == 0.0:
        return np.exp(-np.exp(-x))
    else:
        return np.exp(-1./(1+zeta*x)**(1/zeta))
    
def GEV_inv_cdf(u,zeta):
    if zeta == 0.0:
        return -np.log(-np.log(u))
    else:
        return ((-np.log(u))**(-zeta)-1)/zeta      

# Customize Random Variable
def random_GEV(zeta):
    return GEV_inv_cdf(np.random.uniform(),zeta)

def plot_empirical_cdf(zeta = 1.0):
    GEV_samples = [GEV_inv_cdf(np.random.uniform(),zeta) for _ in range(5000)]
    x_min = np.min(GEV_samples)
    x_max = np.max(GEV_samples)
    x_list = np.linspace(x_min,x_max,100)
    bins = np.linspace(x_min,x_max,50)

    y_true_cdf = GEV_cdf(x_list,zeta)

    plt.hist(GEV_samples, bins, density=True, histtype='step', cumulative=True)
    plt.plot(x_list,y_true_cdf)
    plt.xlim([x_min,x_max])
    plt.show()