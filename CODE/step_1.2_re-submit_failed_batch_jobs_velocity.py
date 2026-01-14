#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import scipy.special as spp
import arviz as av
import pymc as pm
from scipy.stats import skew, kurtosis
import seaborn as sns
sns.set(style='darkgrid')
from utilities_life import ldpkl
import glob
from utils import return_idp_cols


# In[3]:

projdir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity'
data_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data'
folds_dir = os.path.join(data_dir)
python_path = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/envs/test_ptk/bin/python3.10'
normative_path = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/envs/test_ptk/lib/python3.10/site-packages/pcntoolkit/normative.py'
os.chdir(projdir)

#results_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/10_folds_results_long'
data_dir_patch = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Data'

#%%


# In[4]:
features = return_idp_cols()

failed_models = []
columns = features
#%%
failed_models = []

#%%

for l in ['Normal','SHASHo','SHASHb_1','SHASHb_2']:
#for l in ['Normal']:
    print("==============================================================================")
    for fold in range(1):
        print("--------------------------------------------------")
        for i, f in enumerate(features):
            model_path = os.path.join(projdir, f'cross_{fold}_{l}_fit_unscaled_fixed',f'batch_{i+1}','Models',f'NM_0_0_fit.pkl')
            if not os.path.exists(model_path):
                print(f"X Model {l} {f} fold {fold} not found!!!!!!!!!!!!!")
                logdir =  os.path.join(projdir, f'cross_{fold}_{l}_fit_unscaled_fixed','log')
                # print(features.index(f))
                failed_models.append((fold, l, features.index(f), f))
#                 for file in os.listdir(logdir):
#                     if file.startswith(f'fold_{fold}_{l}_{i+1}.sh.e'):
#                         # Store all failed models as tuples
# #                         print(f"fold {fold}, {l} , {i+1}-{f}")
#                         failed_models.append((fold, l, i, f))
            else:
                print(f"Model {l} {f} fold {fold} found!")
#                     /project_cephfs/3022017.02/projects/stijdboe/make_results/10_folds_results/fold_0_SHASHo2/batch_2/Models/NM_0_0_fold0SHASHo2.pkl
for i in failed_models:
    print(i)
    
print(f"{len(failed_models)} models failed")


# - 3-5-2023 - 30 models failed before move - 14 models failed after move
# - 5-5-2023 - 14 models failed before move - 8 models failed after move
# - 8-5-2023 - 8 models failed before move - 7 models failed after move
# - 11-5-2023 - 7 models failed before move - 8 models failed after move
#     - (3, 'SHASHo', 4, 'rh_S_interm_prim-Jensen_thickness')
#     - (4, 'SHASHb_2', 1, 'EstimatedTotalIntraCranialVol')
#     - (4, 'SHASHb_2', 2, 'Right-Lateral-Ventricle')
#     - (4, 'SHASHb_2', 3, 'WM-hypointensities')
#     - (4, 'SHASHb_2', 4, 'rh_S_interm_prim-Jensen_thickness')
#     - (7, 'SHASHb_2', 3, 'WM-hypointensities')
#     - (7, 'SHASHb_2', 5, 'Brain-Stem')
#     - (9, 'SHASHb_2', 4, 'rh_S_interm_prim-Jensen_thickness')
#     
# 

# In[8]:


# del failed_models[1]
# del failed_models[1]

# print(failed_models)
# failed_models = [(0, 'SHASHb_2', 4, 'rh_S_interm_prim-Jensen_thickness')]
# print(failed_models)

# failed_models = [(4, 'SHASHb_2', 1, 'EstimatedTotalIntraCranialVol'),
#                  (4, 'SHASHb_2', 4, 'rh_S_interm_prim-Jensen_thickness'),
#                  (7, 'SHASHb_2', 3, 'WM-hypointensities'),
#                  (7, 'SHASHb_2', 5, 'Brain-Stem'), 
#                  (9, 'SHASHb_2', 4, 'rh_S_interm_prim-Jensen_thickness')]


# In[11]:


batch_size= 1
n_chains= '2'
n_cores_per_batch= '4'  # <- this specifies the number of cores for the job
cores = '4'           # <- this specifies the number of cores for pymc3
n_samples = '1500'
n_tuning='500'
memory = '20gb'
inscaler = 'standardize'
outscaler = 'standardize'

method='bspline'

linear_mu = 'True'
random_intercept_mu='True'
linear_sigma   = 'True'

target_accept = '0.99'
cluster_spec ="torque"


# In[12]:


# For each failed model
model_names = ['SHASHo','SHASHb_1','SHASHb_2','Normal']
likelihood_map = {'SHASHb_1':'SHASHb','SHASHb_2':'SHASHb','SHASHo':'SHASHo','Normal':'Normal'}
durationmap = {'Normal':'20:00:00','SHASHb':'70:00:00','SHASHo':'25:00:00'}
epsilon_linear_map = {'SHASHb_1':'False','SHASHb_2':'True','Normal':'False','SHASHo':'False'}
delta_linear_map = {'SHASHb_1':'False','SHASHb_2':'True','Normal':'False','SHASHo':'False'}


for fold, model_name, feature_i, feature in failed_models:
    likelihood = likelihood_map[model_name]
    duration = durationmap[likelihood]
    linear_epsilon = epsilon_linear_map[model_name]
    linear_delta = delta_linear_map[model_name]
    
    # Unpack the featurename and foldname
    this_identifier = f"patch_fold_{fold}_{model_name}_{feature_i}"
    job_name = this_identifier
    #fold_dir = os.path.join(folds_dir,f'fold_{fold}')
    processing_dir = os.path.join(projdir, this_identifier+'/')
    if not os.path.exists(processing_dir):
        os.mkdir(processing_dir)

    log_dir = os.path.join(processing_dir, 'log')           #
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # The paths to the data
    X_path = os.path.join(data_dir, 'X_train_cross.pkl')
    Y_path = os.path.join(folds_dir, 'Y_train_cross.pkl')
    Z_path = os.path.join(folds_dir,  'trbefile_cross.pkl')
    
    # Check if the fold-feature has been stored already
    patch_fold_feature_data_path = os.path.join(data_dir_patch,'patch_data',f'patch_fold-{fold}_{model_name}_{feature}.pkl')
    
    # Check if the fold-feature has been stored already
    if not os.path.exists(patch_fold_feature_data_path):
        print("Patch data not found")
        with open(Y_path, 'rb') as file:
            Y = pickle.load(file)
            #Y.columns = columns
            Y_patch = Y[feature_i]
        with open(patch_fold_feature_data_path,'wb') as file:
            pickle.dump(Y_patch, file)
    print(fold, likelihood, duration, feature)
    ptk.normative_parallel.execute_nm(processing_dir=processing_dir,
                                          python_path=python_path,
                                          normative_path=normative_path,
                                          job_name = job_name,
                                          n_cores_per_batch = n_cores_per_batch,
                                          cores=cores,
                                          memory=memory,
                                          duration=duration,
                                          batch_size= batch_size,
                                          cluster_spec = cluster_spec,
                                          
                                          savemodel='True',
                                          outputsuffix='fit',
                                          log_path=log_dir,
                                          binary=True,
                                 
                                          covfile_path=X_path,
                                          respfile_path=patch_fold_feature_data_path,
                                          trbefile=Z_path,
        
                                          alg='hbr',
                                          func='fit',
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


# In[5]:


#"make_results/10_folds_results/fold_0_SHASHb_2/batch_1/Models"
#"make_results/10_folds_results/patch_fold_0_SHASHb_2_1/batch_1/Models"
import shutil

for i in os.listdir("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity"):
    if i.startswith('patch'):
        split = i.split("_")
        fold = split[2]
        model = split[3]
        if model == 'SHASHb':
            model = model  + "_"+split[4]
        batch = split[-1]
        batch= int(batch)
        batch = batch +1
        batch = str(batch)
        model_to_copy_path =os.path.join("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity", i, 'batch_1', 'Models', 'NM_0_0_fit.pkl')
        if os.path.exists(model_to_copy_path):
            pass
            print(i + "found")
            target_dest = os.path.join("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/cross_" +fold+ '_' +model+ '_fit_unscaled_fixed', 'batch_'+batch, 'Models', 'NM_0_0_fit.pkl')
            shutil.copyfile(model_to_copy_path, target_dest)
            #print(target_dest)
        else:
            print("X " +i + "not found")
    

# 



