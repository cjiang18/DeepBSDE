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
    exact = 159.89 # set to None when not calculated beforehand (398.03 for r=0.04)

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
                    "y_init_range": [154.37,165.41], #set to None when not sure
                    "num_hiddens": [dim+10, dim+10],
                    "lr_values": [5e-3, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 128,
                    "logging_frequency": 200,
                    "dtype": "float64",
                    "verbose": True
                }
                }

    xva_config = {

                "eqn_config": {
                    "_comment": "xVA estimation",
                    "eqn_name": "XVA",
                    "total_time": total_time,
                    "dim": 1,
                    "num_time_interval": num_time_interval,                    
                    "r":r,
                    "intensityB": 0.01, # default intensity of the bank
                    "intensityC": 0.1, # default intensity of the counterparty
                    "r_fl":r, # unsecured funding lending rate
                    "r_fb":r, # unsecured funding borrowing rate
                    "r_cl":r, # interest rate on posted collateral
                    "r_cb":r, # interest rate on received collateral
                    "collateral":0,
                    "Recovery_Counterparty":0.3, # recovery rate of the bank
                    "Recovery_Bank":0.4 # recovery rate of the counterparty

                    
                },

                "net_config": {
                    "y_init_range": [-10,10], 
                    "num_hiddens": [11, 11],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 128,
                    "logging_frequency": 200,
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
        print('confidence interval is '+str(config.net_config.y_init_range))

    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    #training_history = bsde_solver.train()

    # algotihm 2 & 3
    xva_config = munch.munchify(xva_config)
    xva_bsde = getattr(eqn, xva_config.eqn_config.eqn_name)(xva_config.eqn_config,bsde_solver.model)
    xva_solver = BSDESolver(xva_config,xva_bsde)
    #xva_train_history = xva_solver.train()

    xva_MonteCarlo = xva_bsde.monte_carlo(10000)

    print('Monte Carlo Esimation of xVA: '+str(xva_MonteCarlo))
    print('deep approximation of xVA: '+str(xva_solver.y_init.numpy()))

    
'''
    #estimated epected positive and negative exposure
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))
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
'''
    df = pd.DataFrame(simulations[:,0,:])
    filepath = 'exposureForward' + config.eqn_config.eqn_name + '.xlsx'
    df.to_excel(filepath, index=False)
'''