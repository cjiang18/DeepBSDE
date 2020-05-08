import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
from tqdm import tqdm
import xvaEquation as eqn
import munch


 

if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval=200
    config = {
                "eqn_config": {
                    "_comment": "a forward contract",
                    "eqn_name": "PricingForward",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval
                },
                "net_config": {
                    "y_init_range": [90, 110],
                    "num_hiddens": [dim+20, dim+20],
                    "lr_values": [5e-3, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 100,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()

    #apply trained model to evaluate value of the forward contract via Monte Carlo
    simulations = np.zeros((P,dim,config.eqn_config.num_time_interval+1))
    num_batch = P//batch_size #number of batches
    for i in tqdm(range(num_batch)):
        simulations[i*batch_size:(i+1)*batch_size] = bsde_solver.model.call(bsde.sample(config.net_config.batch_size),False).numpy()
    
    # REMARK: code is only valid when risk-free return r=0
    epe = np.mean(np.maximum(simulations,0),axis=0)
    ene = np.mean(np.minimum(simulations,0),axis=0)


