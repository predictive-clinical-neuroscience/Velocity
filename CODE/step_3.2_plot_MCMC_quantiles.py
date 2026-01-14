# coding: utf-8

# In[1]:

import os
import pandas as pd
from pcntoolkit.util.hbr_utils import *
import numpy as np
import pickle
from matplotlib import pyplot as plt
from pcntoolkit.util.utils import scaler
from matplotlib import patches
cols = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
plt.rc('axes', axisbelow=True)
import gc
gc.collect()
from utils import  DES_idp_cols, sc_idp_cols, DK_idp_cols

#%%
features = DES_idp_cols()

X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_retrain_DES_adults.pkl",'rb')).to_numpy()
Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_retrain_DES_adults.pkl",'rb'))
Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_retrain_DES_adults.pkl",'rb')).to_numpy()
#X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_test_DES_demented_adults.pkl",'rb')).to_numpy()
#Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_test_DES_demented_adults.pkl",'rb'))
#Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/tebefile_DES_demented_adults.pkl",'rb')).to_numpy()
#
# In[4]:
os.chdir('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity')
#%%
def plot_MAP_quantiles(feature ="L_bankssts", l='SHASHo',selected_sex='female', atlas ='DK'):
    f_idx = features.index(feature)

    try:
        model_path = f"{l}_estimate_scaled_fixed_{atlas}_retrain/batch_{f_idx+1}/Models/NM_0_0_estimate.pkl"
        model = pd.read_pickle(model_path)
        print(model.configs)
    

        inscaler = scaler("standardize")
        outscaler = scaler("standardize")
        selected_sex_id = 0 if selected_sex == 'female' else 1
        this_Y_train = Y_train[f_idx].to_numpy()
        this_scaled_X_train = inscaler.fit_transform(X_train)
        this_scaled_Y_train = outscaler.fit_transform(this_Y_train)
        
        #this_Y_test = Y_test[f_idx].to_numpy()
        #this_scaled_X_test = inscaler.transform(X_test)
        #this_scaled_Y_test = outscaler.transform(this_Y_test)
    
        train_sex_idx = np.where(X_train[:,1]==selected_sex_id)
        
        #test_sex_idx = np.where(X_test[:,1]==selected_sex_id)
    
        # select a model batch effect (69,1)
        model_be = [19]
        print(model_be)
        
        mu_intercept_mu = model.hbr.idata.posterior['mu_intercept_mu'].to_numpy().mean()
        sigma_intercept_mu = model.hbr.idata.posterior['sigma_intercept_mu'].to_numpy().mean()
        offsets = model.hbr.idata.posterior['offset_intercept_mu'].to_numpy().mean(axis = (0,1))
        model_offset_intercept_mu_be = offsets[model_be]
    
    
        # Make an empty array
        centered_Y_train = np.zeros_like(this_Y_train)
        #centered_Y_test = np.zeros_like(this_Y_test)
    
        # For each batch effect
        for i, be in enumerate(np.unique(Z_train)):
            this_offset_intercept = offsets[i]
            idx = (Z_train == be).all(1)
    
            centered_Y_train[idx] = this_scaled_Y_train[idx]-sigma_intercept_mu*this_offset_intercept
            #centered_Y_train[idx] = this_Y_train[idx]
            #idx = (Z_test == be).all(1)
            #centered_Y_test[idx] = this_scaled_Y_test[idx]-sigma_intercept_mu*this_offset_intercept
            #centered_Y_test[idx] = this_Y_test[idx]
        fig = plt.figure(figsize=(5,4))
    
        ytrain_inv = outscaler.inverse_transform(centered_Y_train[train_sex_idx,None])
        maxy = np.max(ytrain_inv)
        miny = np.min(ytrain_inv)
        dify = maxy - miny
        #plt.ylim(miny - 0.1*dify, maxy + 0.1*dify )
        #plt.scatter(X_train[train_sex_idx,0], Y_train[train_sex_idx,None], alpha = 0.1, s = 12, color=cols[0])
        #plt.scatter(X_test[test_sex_idx,0], Y_test[test_sex_idx,None], alpha = 0.1, s = 12, color=cols[1])
       
        # all sites
        plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[train_sex_idx,0], outscaler.inverse_transform(centered_Y_train[train_sex_idx,None]), alpha = 0.1, s = 12, color=cols[0])
        #plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[test_sex_idx,0], outscaler.inverse_transform(centered_Y_test[test_sex_idx,None]), alpha = 0.1, s = 12, color=cols[1])
       
        
        be_map =np.unique(Z_train)
        print(be_map[model_be])
        idx = (Z_train == be_map[model_be[0]]).all(1)
        #plt.scatter(X_train[idx,0], centered_Y_train[idx,None], alpha = 0.1, s = 12, color=cols[0])
        #plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[idx,0], outscaler.inverse_transform(centered_Y_train[idx,None]), alpha = 0.1, s = 12, color=cols[0])
      
        #idx = (Z_test == be_map[model_be[0]]).all(1)
        #plt.scatter(X_test[idx,0], centered_Y_test[idx,None], alpha = 0.1, s = 12, color=cols[1])
        #plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[idx,0], outscaler.inverse_transform(centered_Y_test[idx,None]), alpha = 0.1, s = 12, color=cols[1])
       
    
        difX = np.max(this_scaled_X_train[:,0])-np.min(X_train[:,0])
        min0 = np.min(this_scaled_X_train[:,0]) + 0.01*difX
        max0 = np.max(this_scaled_X_train[:,0]) - 0.01*difX
        sex = np.unique(this_scaled_X_train[:,1])[selected_sex_id]
        synthetic_X0 = np.linspace(min0, max0, 200)[:,None]
        #plt.xlim(min0,max0)
        synthetic_X = np.concatenate((synthetic_X0, np.full(synthetic_X0.shape,sex)),axis = 1)
        ran = np.arange(-3,4)
    
        # q = get_single_quantiles(synthetic_X,ran, model, model_be,MAP)-sigma_intercept_mu*offsets[tuple(model_be)]
        model_be_long = np.repeat(np.array(be_map[model_be]),synthetic_X.shape[0])
        #q = model.get_mcmc_quantiles(synthetic_X, model_be_long, z_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]))  - sigma_intercept_mu * offsets[model_be]
        q = model.get_mcmc_quantiles(synthetic_X, model_be_long)  - sigma_intercept_mu * offsets[model_be]
        q = outscaler.inverse_transform(q).T
    
        x = inscaler.inverse_transform(synthetic_X)
        plt.xlim(np.min(x),np.max(x))
        for ir, r in enumerate(ran):
            if r == 0:
                plt.plot(x[:,0], q[:,ir], color = 'black')
            elif abs(r) == 3:
                plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linestyle="--", linewidth = 1)
    
            else:
                plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linewidth = 1)
        lmap = {'blr':'W-BLR','SHASHo':'$\mathcal{S}_o$', 'SHASHb_1':'$\mathcal{S}_{b1}$','SHASHb_2':'$\mathcal{S}_{b2}$',  'Normal':'$\mathcal{N}$'}
    
        suffix = "fit"
        plt.title(lmap[l] + " " + feature, fontsize = 16)
        plt.xlabel('Age',fontsize=15)
    #     plt.ylabel(feature,fontsize=12)
        plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(linestyle=":", linewidth=1, alpha=0.7)
        fig.axes[0].yaxis.offsetText.set_fontsize(14)
        #plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/{l}_estimate_scaled_fixed_{atlas}_demented_adults/imgs/mcmc_quantile_plot_{feature}_{l}_{suffix}_estimate_scaled_fixed_{atlas}_demented_adults.png",bbox_inches='tight',dpi=300)
        #plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/{l}_retrain_{atlas}_adults/imgs/mcmc_quantile_plot_{feature}_{l}_{suffix}_retrain_{atlas}_adults.png",bbox_inches='tight',dpi=300)
        plt.show()
        del model
        del this_Y_train
        #del this_Y_test
        del this_offset_intercept
        #del this_scaled_X_test
        del this_scaled_X_train
        #del this_scaled_Y_test
        del this_scaled_Y_train
        gc.collect()
        # except Exception(e):
        #     print("not found:" + e)
    
    except:
       print("not found")

# In[5]:
for i, feat in enumerate(features):
    if  i == 2:
        break
    plot_MAP_quantiles(l='Normal', feature=feat, atlas ='DES')

