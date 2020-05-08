from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal

class PricingForward(Equation):
    def __init__(self,eqn_config):
        super(PricingForward, self).__init__(eqn_config)
        self.strike = 100
        self.x_init = np.ones(self.dim) * self.strike  # initial value of x, the underlying
        self.sigma = 0.25    # volatility 
        self.mu_bar = 0.0    # drift 
        self.rl = 0.0        # lending rate
        self.rb = 0.0        # borrowing rate      
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
            factor = np.exp((self.mu_bar-(self.sigma**2)/2)*self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   #use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
         	    x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])           
      
        
        return dw_sample, x_sample   

  
    def f_tf(self, t, x, y, z):
        
        temp = tf.reduce_sum(z, 1, keepdims=True) / self.sigma 
        return -self.rl * y - (self.mu_bar - self.rl) * temp  # the driver is  -(mu/sigma)* Z
   
    def g_tf(self, t, x):
        return x - self.strike
    
 