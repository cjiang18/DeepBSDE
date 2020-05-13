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
    r = 0.04
    sigma = 0.25
    x_init = 100
    strike = x_init*dim
    exact = 398.03 # set to None when not calculated beforehand
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
                    "y_init_range": [395, 400],
                    "num_hiddens": [dim+10, dim+10],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
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
    if exact is None:
        print('start \'exact value\' estimation')
        exact = []
        for i in tqdm(range(100)):
            exact+=list(bsde.g_tf(1,bsde.sample(1024)[1][:,:,-1]).numpy()[:,0])
        exact = np.mean(exact)*np.exp(-r)
        print('Exact vlaue is {0:.2f}'.format(exact))

    #apply algorithm 1

    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()  


    #apply trained model to evaluate value of the forward contract via Monte Carlo

    simulations = np.zeros((P,1,config.eqn_config.num_time_interval+1))
    num_batch = P//batch_size #number of batches

    for i in tqdm(range(num_batch)):
         simulations[i*batch_size:(i+1)*batch_size,:,:] = bsde_solver.model.simulate(bsde.sample(config.net_config.batch_size))
    

    #estimated epected positive and negative exposure

    time_stamp = np.linspace(0,1,num_time_interval+1)
    epe = np.mean(np.exp(-r*time_stamp)*np.maximum(simulations,0),axis=0)
    ene = np.mean(np.exp(-r*time_stamp)*np.minimum(simulations,0),axis=0)

    epe_exact = np.array([exact for s in time_stamp])
    ene_exact = np.array([0.0 for s in time_stamp])



    plt.figure()
    plt.plot(time_stamp,epe_exact,'b--',label='DEPE = exact solution')
    plt.plot(time_stamp,epe[0],'b',label='DEPE = deep solver approximation')

    plt.plot(time_stamp,ene_exact,'r--',label='DNPE = exact solution')
    plt.plot(time_stamp,ene[0],'r',label='DNPE = deep solver approximation')

    plt.xlabel('t')
    plt.legend()


    plt.show()   

'''
    df = pd.DataFrame(simulations[:,0,:])
    filepath = 'exposureForward' + config.eqn_config.eqn_name + '.xlsx'
    df.to_excel(filepath, index=False)
'''