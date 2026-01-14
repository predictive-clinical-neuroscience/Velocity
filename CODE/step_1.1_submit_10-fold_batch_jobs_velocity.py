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
from utils import SC_idp_cols, remove_outliers_and_adjust, DES_idp_cols, DK_idp_cols


# In[2]:

projdir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity'

python_path = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/envs/test_ptk/bin/python3.12'
normative_path = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/envs/test_ptk/lib/python3.12/site-packages/pcntoolkit/normative.py'
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
cluster_spec = 'slurm'


# In[4]:
idps = DK_idp_cols()
#measures_name = idps + ["Mean_Thickness", "Median_Thickness"]
#%%
data_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/'

model_names = ['SHASHb_1']
#model_names = ['SHASHb_1']


likelihood_map = {'SHASHb_1':'SHASHb','SHASHb_2':'SHASHb','SHASHo':'SHASHo','Normal':'Normal'}
durationmap = {'Normal':'72:00:00','SHASHb':'72:00:00','SHASHo':'72:00:00'}
epsilon_linear_map = {'SHASHb_1':'False','SHASHb_2':'True','Normal':'False','SHASHo':'False'}
delta_linear_map = {'SHASHb_1':'False','SHASHb_2':'True','Normal':'False','SHASHo':'False'}

for model_name in model_names:
    likelihood = likelihood_map[model_name]
    duration = durationmap[likelihood]
    linear_epsilon = epsilon_linear_map[model_name]
    linear_delta = delta_linear_map[model_name]
    
    
    X_train_path_full = os.path.join(data_dir, 'X_train_retrain_DK_adults_ADNI.pkl')
    Y_train_path_full = os.path.join(data_dir, 'Y_train_retrain_DK_adults_ADNI.pkl')
    Z_train_path_full = os.path.join(data_dir, 'trbefile_retrain_DK_adults_ADNI.pkl')
   
    
    X_test_path_full = os.path.join(data_dir, 'X_train_retrain_DK_adults_ADNI.pkl')
    Y_test_path_full = os.path.join(data_dir, 'Y_train_retrain_DK_adults_ADNI.pkl')
    Z_test_path_full = os.path.join(data_dir, 'trbefile_retrain_DK_adults_ADNI.pkl')
  
    #X_train_full= ldpkl(X_train_path_full)
    #sum(X_train.isna())

    #Z_train_full = ldpkl(Z_train_path_full)   
    #sum(Z_train.isna())

    #Y_train_full = ldpkl(Y_train_path_full)  
    
    # for i, idp in enumerate(idps):
        
    #this_identifier = f"{model_name}_estimate_scaled_fixed_DES_retrain_ADNI"
    this_identifier = f"{model_name}_estimate_scaled_fixed_DK_retrain_ADNI"
    job_name = this_identifier
    #fold_dir = os.path.join(folds_dir,f'fold_{i}')
    processing_dir = os.path.join(projdir,this_identifier+'/')
    if not os.path.exists(processing_dir):
        os.mkdir(processing_dir)

    log_dir = os.path.join(processing_dir, 'log/')           #
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

        # The paths to the data
        #X_path = os.path.join(folds_dir, f'fold_{i}', 'X_train.pkl')
        #Y_path = os.path.join(folds_dir, f'fold_{i}', 'Y_train.pkl')
        #Z_path = os.path.join(folds_dir, f'fold_{i}', 'trbefile.pkl')
        #Z_path = os.path.join(folds_dir, f'fold_{i}', 'Z_train.pkl')
        
        #cross
        #X_train_path = os.path.join(data_dir, 'X_train_cross_DK.pkl')
        #Y_train_path = os.path.join(data_dir, 'Y_train_cross_DK.pkl')
       # Z_train_path = os.path.join(data_dir, 'trbefile_cross_DK.pkl')
       
       # X_test_path = os.path.join(data_dir, 'X_test_cross_DK.pkl')
        #Y_test_path = os.path.join(data_dir, 'Y_test_cross_DK.pkl')
       # Z_test_path = os.path.join(data_dir, 'tebefile_cross_DK.pkl')
       
       # both cortical thickness
        #X_train_path = os.path.join(data_dir, 'X_train_retrain_DES_adults.pkl')
        #Y_train_path = os.path.join(data_dir, 'Y_train_retrain_DES_adults.pkl')
        #Z_train_path = os.path.join(data_dir, 'trbefile_retrain_DES_adults.pkl')
      
        #X_test_path = os.path.join(data_dir, 'X_train_retrain_DES_adults.pkl') #includes demented
        #Y_test_path = os.path.join(data_dir, 'Y_train_retrain_DES_adults.pkl')
        #Z_test_path = os.path.join(data_dir, 'trbefile_retrain_DES_adults.pkl')
        
        
        #X_train_path = os.path.join(data_dir, 'X_train_DK_demented_adults_ADNI.pkl')
        #Y_train_path = os.path.join(data_dir, 'Y_train_DK_demented_adults_ADNI.pkl')
        #Z_train_path = os.path.join(data_dir, 'trbefile_DK_demented_adults_ADNI.pkl')
      
        #X_test_path = os.path.join(data_dir, 'X_test_DK_demented_adults_ADNI.pkl')
        #Y_test_path = os.path.join(data_dir, 'Y_test_DK_demented_adults_ADNI.pkl')
        #Z_test_path = os.path.join(data_dir, 'tebefile_DK_demented_adults_ADNI.pkl')

        
        #X_train_path = os.path.join(data_dir, 'X_train_retrain_DES_adults_ADNI.pkl')
        #Y_train_path = os.path.join(data_dir, 'Y_train_retrain_DES_adults_ADNI.pkl')
        #Z_train_path = os.path.join(data_dir, 'trbefile_retrain_DES_adults_ADNI.pkl')
      
        #X_test_path = os.path.join(data_dir, 'X_train_retrain_DES_adults_ADNI.pkl')
        #Y_test_path = os.path.join(data_dir, 'Y_train_retrain_DES_adults_ADNI.pkl')
        #Z_test_path = os.path.join(data_dir, 'trbefile_retrain_DES_adults_ADNI.pkl') 
        #sum(Y_train.isna())
        
        #os.chdir(projdir)
        #Y_train = Y_train_full[i]
        
        #X_train, Y_train, Z_train = remove_outliers_and_adjust(X_train_full, Y_train, Z_train_full)
        
       # X_train_path = f"{projdir}/{this_identifier}/X_train_retrain_sc_{idp}.pkl"
       # Y_train_path = f"{projdir}/{this_identifier}/Y_train_retrain_sc_{idp}.pkl"
        #Z_train_path = f"{projdir}/{this_identifier}/Z_train_retrain_sc_{idp}.pkl"
    
        # with open(X_train_path, 'wb') as file:
        #     pickle.dump(pd.DataFrame(X_train), file)  
        # with open(Y_train_path, 'wb') as file:
        #     pickle.dump(pd.DataFrame(Y_train), file)
        # with open(Z_train_path, 'wb') as file:
        #     pickle.dump(pd.DataFrame(Z_train), file)

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
                             
                                      covfile_path=X_train_path_full,
                                      respfile_path=Y_train_path_full,
                                      trbefile=Z_train_path_full,
                                      
                                      testcovfile_path=X_test_path_full,
                                      testrespfile_path=Y_test_path_full,
                                      tsbefile=Z_test_path_full,
                                      
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
# predict
model_name = 'Normal'
likelihood = likelihood_map[model_name]
duration = durationmap[likelihood]
linear_epsilon = epsilon_linear_map[model_name]
linear_delta = delta_linear_map[model_name]

nm_func = 'predict'
outputsuffix = '_' + nm_func


#testrespfile = data_dir + 'Y_test_OASIS23_long_demented.pkl'
#testcovfile = data_dir + 'X_test_OASIS23_long_demented.pkl'
#tsbefile = data_dir + 'tebefile_OASIS23_long_demented.pkl'
testrespfile = os.path.join(data_dir, 'Y_test_DES_demented_adults.pkl')
testcovfile = os.path.join(data_dir, 'X_test_DES_demented_adults.pkl')
tsbefile = os.path.join(data_dir, 'tebefile_DES_demented_adults.pkl')


job_name = "Normal_estimate_scaled_fixed_DES_demented_adults"

#%%
processing_dir = os.path.join(projdir,job_name+'/')
if not os.path.exists(processing_dir):
    os.mkdir(processing_dir)

log_dir = os.path.join(processing_dir, 'log')           #
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
#%%
ptk.normative_parallel.execute_nm(processing_dir, 
                                  python_path=python_path,
                                  normative_path=normative_path,
                                  job_name = job_name,
                                  batch_size= batch_size,
                                  n_cores_per_batch = n_cores_per_batch,
                                  cores=cores,
                                  memory=memory,
                                  duration=duration,
                     
                                  alg='hbr', 
                                  log_path=log_dir, 
                                  func=nm_func, 
                                  savemodel='True', 
                                  inputsuffix='_estimate',
                                  outputsuffix=outputsuffix, 
                                  inscaler=inscaler, 
                                  outscaler=outscaler,
                                  cluster_spec='slurm',
                                
                                  binary=True,

                                  covfile_path=testcovfile,
                                  respfile_path=testrespfile,
                                  tsbefile=tsbefile,
                                
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
processing_dir='/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_retrain_ADNI/'
nm_func = "estimate"
batch_size = 1
outputsuffix ="_estimate"
job_name = "estimate"

#%%
memory = '20gb'
duration = '72:00:00'
log_dir = processing_dir + '/log/'
cluster_spec = 'slurm'
cores = '4'

#%%
ptk.normative_parallel.collect_nm(processing_dir, 
                                  job_name =job_name,
                                  func=nm_func, 
                                  collect=True, 
                                  binary=True, 
                                  batch_size=batch_size, 
                                  outputsuffix=outputsuffix)

#%%
ptk.normative_parallel.sbatchrerun_nm(processing_dir, 
                                log_dir, 
                                memory,
                                cluster_spec,
                                cores,
                                duration, 
                                #binary=True, 
                                interactive=True)

#%%

pkl_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Normal_estimate_scaled_fixed_DK_demented_adults/failed_batches.pkl"
with open(pkl_path, "rb") as f:
    script_paths = pickle.load(f)
#%%
for script in script_paths:
    try:
        result = subprocess.run(["sbatch", script], capture_output=True, text=True, check=True)
        print(f"Submitted: {script} → {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit: {script}")
        print(f"Error: {e.stderr.strip()}")    
#%%
# Load the files
with open(X_train_path, 'rb') as f:
    X_train = pickle.load(f)

with open(Y_train_path, 'rb') as f:
    Y_train = pickle.load(f)

with open(Z_train_path, 'rb') as f:
    Z_train = pickle.load(f)

with open(X_test_path, 'rb') as f:
    X_test = pickle.load(f)

with open(Y_test_path, 'rb') as f:
    Y_test = pickle.load(f)

with open(Z_test_path, 'rb') as f:
    Z_test = pickle.load(f)