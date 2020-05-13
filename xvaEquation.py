from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal

class PricingForward(Equation):
    def __init__(self,eqn_config):
        super(PricingForward, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x, the underlying
        self.sigma = eqn_config.sigma    # volatility 
        self.rate = eqn_config.r    # drift     
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
        return -self.rate*y
   
    def g_tf(self, t, x):
        return x - self.strike



    
class BasketOption(Equation):

    def __init__(self, eqn_config):

        super(BasketOption, self).__init__(eqn_config)
        self.x_init = eqn_config.x_init
        self.strike = eqn_config.strike
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
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
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]

        else:   #use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
         	    x_sample[:, :, i + 1] = (1 + self.r * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])           
    

        return dw_sample, x_sample   



    def f_tf(self, t, x, y, z):

        return -self.r * y


    def g_tf(self, t, x):

        temp = tf.reduce_sum(x, 1,keepdims=True)
        return tf.maximum(temp - self.strike, 0)