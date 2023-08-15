import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="100"

import pickle
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

import numpy as np
import matplotlib.pyplot as plt

from time import time
from pinn_utils import *
from multiprocessing import Pool



def run(trial):
    # Model keyword
    key = 'ADAF' # or 'M2' or 'R'
    num_hidden_layers=4
    num_neurons_per_layer=20

    # Material Properties
    xmin = -0.5
    xmax = 1.0

    ymin = -0.5
    ymax = 1.5


    # Properties_dict
    properties = {
        'xmin':xmin,
        'xmax':xmax,
        'xmin':ymin,
        'xmax':ymax,    
        }

    DTYPE = 'float32'
    # Set number of data points
    N_b = 101
    N_r = 2601

    #Model construction
    lb = tf.constant([xmin, ymin], dtype=DTYPE)
    ub = tf.constant([xmax, ymax], dtype=DTYPE)
    pinn = Build_PINN(lb, ub, properties, num_hidden_layers, num_neurons_per_layer, key)
    pinn.model.summary()

    #Solver
    solver = Solver_PINN(pinn, properties, N_b=N_b, N_r=N_r)
    #Train Adam
    ref_time = time()
    solver.train_adam(200)
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    #Train lbfgs
    ref_time = time()
    solver.ScipyOptimizer(method='L-BFGS-B', 
        options={'maxiter': 40000, 
            'maxfun': 50000, 
            'maxcor': 50, 
            'maxls': 50, 
            'ftol': np.finfo(float).eps,
            'gtol': np.finfo(float).eps,            
            'factr':np.finfo(float).eps,
            'iprint':50})
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    solver.save_results(trial)     

if __name__== '__main__':
    p = Pool(processes=10)
    p.map(run, list(range(10)))   