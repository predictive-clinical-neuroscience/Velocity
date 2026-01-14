#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pcntoolkit as ptk
from itertools import product
from functools import reduce
from pcntoolkit.model.SHASH import SHASHb, SHASH, SHASHo
from scipy.stats import gaussian_kde
import pymc as pm
import scipy.special as spp
import arviz as av
from scipy.stats import skew, kurtosis
#import seaborn as sns
#sns.set(style='darkgrid')
from utils import ldpkl


# In[2]:


projdir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity'
data_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data'
#data_dir = '/project_cephfs/3022017.06/projects/stijdboe/Data'
folds_dir = os.path.join(data_dir)
python_path = '/home/preclineu/johbay/.conda/envs/py310/bin/python3.10'
normative_path = '/home/preclineu/johbay/.conda/envs/py310/lib/python3.10/site-packages/pcntoolkit/normative.py'
os.chdir(projdir)

# In[3]:


batch_size= 1
n_chains= '2'
n_cores_per_batch= '4'  # <- this specifies the number of cores for the job
cores = '4'           # <- this specifies the number of cores for pymc3
n_samples = '1000'
n_tuning='500'
memory = '20gb'
inscaler = 'standardize'
outscaler = 'standardize'
method='bspline'
linear_mu = 'True'
random_intercept_mu='True'
linear_sigma   = 'True'
target_accept = '0.99'
#cluster_spec = 'sbatch'
cluster_spec = 'torque'


# In[4]:

model_names = ['Normal', 'SHASHo', 'SHASHb_1']
#model_names = ['Normal', 'SHASHb_1','SHASHb_2', 'SHASHo']
#model_names = ['Normal']

likelihood_map = {'SHASHb_1':'SHASHb','SHASHb_2':'SHASHb','SHASHo':'SHASHo','Normal':'Normal'}
durationmap = {'Normal':'72:00:00','SHASHb':'72:00:00','SHASHo':'72:00:00'}
epsilon_linear_map = {'SHASHb_1':'False','SHASHb_2':'True','Normal':'False','SHASHo':'False'}
delta_linear_map = {'SHASHb_1':'False','SHASHb_2':'True','Normal':'False','SHASHo':'False'}

for model_name in model_names:
    likelihood = likelihood_map[model_name]
    duration = durationmap[likelihood]
    linear_epsilon = epsilon_linear_map[model_name]
    linear_delta = delta_linear_map[model_name]
    
    for i in range(0,1):
        
        this_identifier = f"train-long_test-long_estimate_scaled_fixed_sc_{model_name}"
        #this_identifier = f"{i}_{model_name}_estimate_testkit"
        job_name = this_identifier
        #fold_dir = os.path.join(folds_dir,f'fold_{i}')
        processing_dir = os.path.join(projdir,this_identifier+'/')
        if not os.path.exists(processing_dir):
            os.mkdir(processing_dir)

        log_dir = os.path.join(processing_dir, 'log')           #
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        # The paths to the data
        
        
        #X_train_path = os.path.join(data_dir, 'X_train_cross_sc.pkl')
        #Y_train_path = os.path.join(data_dir, 'Y_train_cross_sc.pkl')
        #Z_train_path = os.path.join(data_dir, 'trbefile_cross_sc.pkl')
      
        #X_test_path = os.path.join(data_dir, 'X_test_cross_sc.pkl')
        #Y_test_path = os.path.join(data_dir, 'Y_test_cross_sc.pkl')
        #Z_test_path = os.path.join(data_dir, 'tebefile_cross_sc.pkl')
    
        X_train_path = os.path.join(data_dir, 'X_train_long_sc.pkl')
        Y_train_path = os.path.join(data_dir, 'Y_train_long_sc.pkl')
        Z_train_path = os.path.join(data_dir, 'trbefile_long_sc.pkl')
  
        X_test_path = os.path.join(data_dir, 'X_test_long_sc.pkl')
        Y_test_path = os.path.join(data_dir, 'Y_test_long_sc.pkl')
        Z_test_path = os.path.join(data_dir, 'tebefile_long_sc.pkl')

        

        ptk.normative_parallel.execute_nm(processing_dir=processing_dir,
                                          python_path=python_path,
                                          normative_path=normative_path,
                                          job_name = job_name,
                                          n_cores_per_batch = n_cores_per_batch,
                                          cores=cores,
                                          memory=memory,
                                          duration=duration,
                                          batch_size= batch_size,
                                          
                                          
                                          savemodel='True',
                                          outputsuffix='estimate',
                                          log_path=log_dir,
                                          binary=True,
                                          cluster_spec=cluster_spec,
    
                                 
                                          covfile_path=X_train_path,
                                          respfile_path=Y_train_path,
                                          trbefile=Z_train_path,
                                          
                                          testcovfile_path=X_test_path,
                                          testrespfile_path=Y_test_path,
                                          tsbefile=Z_test_path,
                                          
                                          alg='hbr',
                                          func='estimate',
                                          inscaler=inscaler,
                                          outscaler=outscaler, 
                                          model_type=method,
 
                                          likelihood = likelihood,
                                          linear_mu=linear_mu,
                                          random_intercept_mu=random_intercept_mu,
                                          random_slope_mu = 'False',
                                          random_sigma='False',
                                          random_intercept_sigma='False',
                                          random_slope_sigma='False',
                                          linear_sigma=linear_sigma,
                                          linear_epsilon=linear_epsilon,
                                          linear_delta=linear_delta,
                                          target_accept = target_accept,
                                          
                                          n_samples=n_samples,
                                          n_tuning=n_tuning,
                                          n_chains=n_chains,
                                          interactive=False)


#%%
X_train= ldpkl(X_train_path)
sum(X_train.isna())

Z_train = ldpkl(Z_train_path)   
sum(Z_train.isna())

Y_train = ldpkl(Y_train_path)   
sum(Y_train.isna())

X_test= ldpkl(X_test_path)
sum(X_test.isna())

Z_test = ldpkl(Z_test_path)   
sum(Z_test.isna())

Y_test = ldpkl(Y_test_path)   
sum(Y_test.isna())

#%%

test = ldpkl("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/0_Normal_estimate_scaled_fixed_DK_CT/batch_1/Z_estimate.pkl")
plt.scatter(X_test[0], Y_test[0])
#plt.ylim(-10,5)
plt.show()
