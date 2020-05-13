from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from tqdm import tqdm

class EuropeanEquation(Equation):
    # abstract class for European type xVA evaluation
    def __init__(self,eqn_config):
        super(EuropeanEquation,self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init =  eqn_config.x_init  # initial value of x, the underlying
        self.sigma = eqn_config.sigma    # volatility 

        try:
            self.rate = eqn_config.r    # risk-free rate 
        except AttributeError:
            self.rate = 0.0   # when not specified, set to 0

        self.useExplict = False #whether to use explict formula to evaluate dyanamics of x

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: #use analytic solution of linear SDE
            factor = np.exp((self.rate-(self.sigma**2)/2)*self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   #use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
         	    x_sample[:, :, i + 1] = (1 + self.rate * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])       
        
        return dw_sample, x_sample  


    def f_tf(self, t, x, y, z):

        return -self.rate * y


    def y_init_estimate(self,num_sample=1024):
        print('automatic y_init_range estimation started')
        estimate = []           
        for i in tqdm(range(num_sample//1024)): #split into batches to estimate
            estimate += list(self.g_tf(1,self.sample(1024)[1][:,:,-1]).numpy()[:,0])
        if num_sample%1024!= 0: #calculate the reminder number of smaples
            estimate += list(self.g_tf(1,self.sample(num_sample%1024)[1][:,:,-1]).numpy()[:,0]) 
        estimate = np.array(estimate)*np.exp(-self.rate*self.total_time)
        mean = np.mean(estimate)
        std = np.std(estimate)/np.sqrt(num_sample)

        return mean, [mean-3*std,mean+3*std]


class PricingForward(EuropeanEquation):
    def __init__(self,eqn_config):
        super(PricingForward, self).__init__(eqn_config)     
  

    def g_tf(self, t, x):
        return x - self.strike


    
class BasketOption(EuropeanEquation):

    def __init__(self, eqn_config):
        super(BasketOption, self).__init__(eqn_config)
      


    def g_tf(self, t, x):

        temp = tf.reduce_sum(x, 1,keepdims=True)
        return tf.maximum(temp - self.strike, 0)