import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
from tqdm import tqdm
import xvaEquation as eqn
import munch
import pandas as pd



if __name__ == "__main__":

    dim = 100 #dimension of brownian motion
    P = 2048 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval = 100
    r = 0.01
    sigma = 0.25
    x_init = 100
    strike = x_init*dim
    exact = None # set to None when not calculated beforehand (398.03 for r=0.04)

    config = {

                "eqn_config": {
                    "_comment": "a basket call option",
                    "eqn_name": "BasketOption",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "strike":strike,
                    "r":r,
                    "sigma":sigma,
                    "x_init":x_init
                },

                "net_config": {
                    "y_init_range": [397, 399], #set to None when not sure
                    "num_hiddens": [dim+10, dim+10],
                    "lr_values": [5e-3, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 2000,
                    "batch_size": batch_size,
                    "valid_size": 128,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }

    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)   

    #estimate the 'exact' solution via Monte Carlo Simulation
    if exact is None or config.net_config.y_init_range is None:
        exact,config.net_config.y_init_range = bsde.y_init_estimate(int(1e4))
        print('Exact vlaue is {0:.2f}'.format(exact))

    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()    
    

    #estimated epected positive and negative exposure
    simulations = bsde_solver.model.predict(bsde.sample(P))
    time_stamp = np.linspace(0,1,num_time_interval+1)
    epe = np.mean(np.exp(-r*time_stamp)*np.maximum(simulations,0),axis=0)
    ene = np.mean(np.exp(-r*time_stamp)*np.minimum(simulations,0),axis=0)

    epe_exact = np.array([exact for s in time_stamp])
    ene_exact = np.array([0.0 for s in time_stamp])


    plt.figure()
    plt.plot(time_stamp,epe_exact,'b--',label='DEPE = exact solution')
    plt.plot(time_stamp,epe[0],'b',label='DEPE = deep solver approximation')

    #plt.plot(time_stamp,ene_exact,'r--',label='DNPE = exact solution')
    #plt.plot(time_stamp,ene[0],'r',label='DNPE = deep solver approximation')

    plt.xlabel('t')
    plt.legend()


    plt.show()   

'''
    df = pd.DataFrame(simulations[:,0,:])
    filepath = 'exposureForward' + config.eqn_config.eqn_name + '.xlsx'
    df.to_excel(filepath, index=False)
'''