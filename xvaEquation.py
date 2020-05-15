from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from tqdm import tqdm

class EuropeanEquation(Equation):
    # abstract class for European type xVA evaluation. 
    # Function g_tf has to be implemented
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
        if num_sample%1024!= 0: #calculate the remaining number of smaples
            estimate += list(self.g_tf(1,self.sample(num_sample%1024)[1][:,:,-1]).numpy()[:,0]) 
        estimate = np.array(estimate)*np.exp(-self.rate*self.total_time)
        mean = np.mean(estimate)
        std = np.std(estimate)/np.sqrt(num_sample)

        return mean, [mean-3*std,mean+3*std]


class ForwardContract(EuropeanEquation):
    def __init__(self,eqn_config):
        super(ForwardContract, self).__init__(eqn_config)     
  

    def g_tf(self, t, x):
        return x - self.strike


    
class BasketOption(EuropeanEquation):

    def __init__(self, eqn_config):
        super(BasketOption, self).__init__(eqn_config)     


    def g_tf(self, t, x):
        temp = tf.reduce_sum(x, 1,keepdims=True)
        return tf.maximum(temp - self.strike, 0)

class Portfolio():

    #!!!!!not yet finished!!!!
    
    def __init__(self,assets,weights=None):
        # assets: a list of NonsharedModel
        # weights: a list of weights for each asset. When not given, assumes equal weights  
        # !!!Warning!!! Assumes independence between random drivers of each asset    
        self.assets = assets
        self.num_assets = len(assets)
        if weights is None:
            self.weights = [1/self.num_assets for i in range(self.num_assets)]
        else:
            self.weights = weights

    def sample(self, num_sample):
        #dw: averaged brownian drive
        #dw: averaged 
        dw,dv = self.assets[0].predict(self.assets[0].bsde.sample(num_sample))
        if self.num_assets>1:
            dw = dw*self.weights[0]
            dv = dv*self.weights[0]
            for i, weight in enumerate(self.weights[1:]):
                dw_,dv_=self.assets[i+1].sample(self.assets[0].bsde.sample(num_sample))
                dw = dw + dw_*weight
                dv = dv + dv_*weight
        return dw,dv



class XVA(Equation):
    def __init__(self,eqn_config,clean_value):
        super(XVA,self).__init__(eqn_config)        
        self.rate = eqn_config.r # risk-free return
        self.intensityB = eqn_config.intensityB # default intensity of the bank
        self.intensityC = eqn_config.intensityC # default intensity of the counterparty
        self.r_fl = eqn_config.r_fl # unsecured funding lending rate
        self.r_fb = eqn_config.r_fb # unsecured funding borrowing rate
        self.r_cl = eqn_config.r_cl # interest rate on posted collateral
        self.r_cb = eqn_config.r_cb # interest rate on received collateral
        self.clean_value = clean_value  # Class NonsharedModel, respresenting the clean value process
        self.collateral = eqn_config.collateral
        self.isXVA = True #indicating that we are dealing with x

        try:
            self.num_assets = clean_value.num_assets # when clean_value is a of class Portfolio
        except AttributeError:
            self.num_assets = None # when clean_value is a of class NonsharedModel, set to None

        #setting (default) value for recovery rate
        try:
            self.R_B = eqn_config.Recovery_Bank
        except AttributeError:
            self.R_B = 0.6
        try:
            self.R_C = eqn_config.Recovery_Counterparty
        except AttributeError:
            self.R_C = 0.6

    def sample(self, num_sample):
        if self.num_assets is None:
            return self.clean_value.predict(self.clean_value.bsde.sample(num_sample))
        else:
            return self.clean_value.sample(num_sample)

    def f_tf(self, t, x, y, z,v_clean):
        # x: clean value, y:xva, z:control
        cva = (1-self.R_C)*tf.maximum(self.collateral-v_clean,0)*self.intensityC
        dva = (1-self.R_B)*tf.maximum(v_clean-self.collateral,0)*self.intensityB
        fva = (self.r_fl-self.rate)*tf.maximum(v_clean-y-self.collateral,0) - (self.r_fb-self.rate)*tf.maximum(self.collateral+y-v_clean,0)
        colva = (self.r_cl-self.rate)*max(self.collateral,0) - (self.r_cb-self.rate)*max(-self.collateral,0)
        discount = -(self.rate+self.intensityB+self.intensityC)*y

        return -cva + dva + fva + colva + discount
    
    def g_tf(self, t, x):
        return 0.

    def monte_carlo(self,num_sample=1024): #monte carlo estimation of CVA and DVA
        
        print('CVA and DVA Monte Carlo estimation started')
        discount = np.exp(-(self.rate+self.intensityB+self.intensityC)*np.linspace(0,self.total_time,self.num_time_interval+1))
       
        estimate = []           
        for i in tqdm(range(num_sample//1024)): #split into batches to estimate
            v = self.sample(1024)[1] #shape (num_sample, dim=1, num_time_interval+1) 
            phi_cva = (1-self.R_C)*discount*np.maximum(self.collateral-v,0)*self.intensityC
            phi_dva = (1-self.R_B)*discount*np.maximum(v-self.collateral,0)*self.intensityB

            #trapeziodal rule
            cva = np.sum(phi_cva,axis=-1)-(phi_cva[:,:,-1]+phi_cva[:,:,0])/2
            dva = np.sum(phi_dva,axis=-1)-(phi_dva[:,:,-1]+phi_dva[:,:,0])/2

            estimate += list(dva[:,0]-cva[:,0])


        if num_sample%1024!= 0: #calculate the remaining number of smaples
            v = self.sample(num_sample%1024)[1] #shape (num_sample, dim=1, num_time_interval+1) 
            phi_cva = (1-self.R_C)*discount*np.maximum(self.collateral-v,0)*self.intensityC
            phi_dva = (1-self.R_B)*discount*np.maximum(v-self.collateral,0)*self.intensityB

            #trapeziodal rule
            cva = np.sum(phi_cva,axis=-1)-(phi_cva[:,:,-1]+phi_cva[:,:,0])/2
            dva = np.sum(phi_dva,axis=-1)-(phi_dva[:,:,-1]+phi_dva[:,:,0])/2

            estimate += list(dva[:,0]-cva[:,0])
        estimate = np.array(estimate)*self.delta_t #times time-interval (height of a trapezium)
        mean = np.mean(estimate)
        std = np.std(estimate)/np.sqrt(num_sample)

        return mean, [mean-3*std,mean+3*std]