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
from utils import  DES_idp_cols, SC_idp_cols, DK_idp_cols



# In[3]:

data_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimated_scaled_fixed_SC_demented_adults_ADNI/'


# Get data


#X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_SC_plot_subjects.pkl",'rb')).to_numpy()
#Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_SC_plot_subjects.pkl",'rb'))
#Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_SC_plot_subject.pkl",'rb')).to_numpy()

atlas="SC"

Z_train_large = pd.read_pickle(f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_retrain_{atlas}_adults.pkl')
X_train_large = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_retrain_{atlas}_adults.pkl",'rb')).to_numpy()
Y_train_large = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_retrain_{atlas}_adults.pkl",'rb'))

#lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_013_S_1186_1.pkl')
#lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_013_S_1186_6.pkl')

#lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_002_S_1070_2.pkl')
#data = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df5_002_S_1070.pkl')

#data = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df4_018_S_0142.pkl')
#lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_018_S_0142_1.pkl')
#lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_018_S_0142_6.pkl')

data = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df1_099_S_0051.pkl')
#lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_099_S_0051_1.pkl')
lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_099_S_0051_4.pkl')
#data = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df2_013_S_1186.pkl')
#lines = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_013_S_1186_1.pkl')
#data2 = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/data_5_subjects.pkl')
#data = data2
# In[4]:
os.chdir('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity')
#%%
def plot_MAP_quantiles(feature ="L_bankssts", l='SHASHo',selected_sex='female', atlas ='SC'):
    f_idx = features.index(feature)

    #try:
    model_path = f"{l}_estimate_scaled_fixed_{atlas}_retrain_ADNI/batch_{f_idx+1}/Models/NM_0_0_estimate.pkl"
    model = pd.read_pickle(model_path)
    print(model.configs)


    inscaler = scaler("standardize")
    outscaler = scaler("standardize")
    selected_sex_id = 0 if selected_sex == 'female' else 1
    this_Y_train = Y_train_large[f_idx].to_numpy()
    #this_Y_train = Y_train[f_idx]
    this_scaled_X_train = inscaler.fit_transform(X_train_large)
    this_scaled_Y_train = outscaler.fit_transform(this_Y_train)
    
    #this_Y_test = Y_test[f_idx].to_numpy()
    #this_Y_test = Y_test[f_idx]
    #this_scaled_X_test = inscaler.transform(X_test)
    #this_scaled_Y_test = outscaler.transform(this_Y_test)

    train_sex_idx = np.where(X_train_large[:,1]==selected_sex_id)
    
    #test_sex_idx = np.where(X_test[:,1]==selected_sex_id)

    # select a model batch effect (69,1)

    model_be = [3]
    print(model_be)
    mu_intercept_mu = model.hbr.idata.posterior['mu_intercept_mu'].to_numpy().mean()
    sigma_intercept_mu = model.hbr.idata.posterior['sigma_intercept_mu'].to_numpy().mean()
    offsets = model.hbr.idata.posterior['offset_intercept_mu'].to_numpy().mean(axis = (0,1))
    model_offset_intercept_mu_be = offsets[model_be]


    # Make an empty array
    centered_Y_train = np.zeros_like(this_Y_train)
    #centered_Y_test = np.zeros_like(this_Y_test)

    # For each batch effect
    for i, be in enumerate(np.unique(Z_train_large)):
        this_offset_intercept = offsets[i]
        idx = (Z_train_large == be).all(1)

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
    #plt.scatter(X_train[:,0], Y_train[:,None], alpha = 0.1, s = 12, color=cols[0])
   
    #plt.scatter(X_test[test_sex_idx,0], Y_test[test_sex_idx,None], alpha = 0.1, s = 12, color=cols[1])
   
    # all sites
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[train_sex_idx,0], outscaler.inverse_transform(centered_Y_train[train_sex_idx,None]), alpha = 0.1, s = 12, color=cols[0])
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[test_sex_idx,0], outscaler.inverse_transform(centered_Y_test[test_sex_idx,None]), alpha = 0.1, s = 12, color=cols[1])
   
    
    be_map =np.unique(Z_train_large)
    print(be_map[model_be])
    idx = (Z_train_large == be_map[model_be[0]]).all(1)
    #plt.scatter(X_train[idx,0], centered_Y_train[idx,None], alpha = 0.1, s = 12, color=cols[0])
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[idx,0], outscaler.inverse_transform(centered_Y_train[idx,None]), alpha = 0.1, s = 12, color=cols[0])
  
    #idx = (Z_test == be_map[model_be[0]]).all(1)
    #plt.scatter(X_test[idx,0], centered_Y_test[idx,None], alpha = 0.1, s = 12, color=cols[1])
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[idx,0], outscaler.inverse_transform(centered_Y_test[idx,None]), alpha = 0.1, s = 12, color=cols[1])
   

    difX = np.max(this_scaled_X_train[:,0])-np.min(X_train_large[:,0])
    #min0 = np.min(this_scaled_X_train[:,0]) + 0.01*difX
    #max0 = np.max(this_scaled_X_train[:,0]) - 0.01*difX
    min0 = np.min(this_scaled_X_train[:,0]) + 0.1*difX
    max0 = np.max(this_scaled_X_train[:,0]) - 0.1*difX
    sex = np.unique(this_scaled_X_train[:,1])[selected_sex_id]
    synthetic_X0 = np.linspace(min0, max0, 100)[:,None]
    #plt.xlim(min0,max0)
    synthetic_X = np.concatenate((synthetic_X0, np.full(synthetic_X0.shape,sex)),axis = 1)
    ran = np.arange(-3,4)

    # q = get_single_quantiles(synthetic_X,ran, model, model_be,MAP)-sigma_intercept_mu*offsets[tuple(model_be)]
    model_be_long = np.repeat(np.array(be_map[model_be]),synthetic_X.shape[0])
    #q = model.get_mcmc_quantiles(synthetic_X, model_be_long, z_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]))  - sigma_intercept_mu * offsets[model_be]
    q = model.get_mcmc_quantiles(synthetic_X, model_be_long)  - sigma_intercept_mu * offsets[model_be]
    q = outscaler.inverse_transform(q).T

    x = inscaler.inverse_transform(synthetic_X)
    
    
    z_single = lines["y"]
    start1, end1 = z_single.iloc[0], z_single.iloc[1]
    start2, end2 = z_single.iloc[2], z_single.iloc[3]

    vals1 = np.linspace(start1, end1, 10)
    vals2 = np.linspace(start2, end2, 10)

    z_single = pd.DataFrame({"y": np.concatenate([vals1, vals2])})
    

    x_single = np.stack((lines["x"], np.full(lines.shape[0], selected_sex_id)),axis = 1)
    
    start1, end1 = x_single[0,0], x_single[1,0]
    start2, end2 = x_single[2,0], x_single[3,0]

    vals1 = np.linspace(start1, end1, 10)
    vals2 = np.linspace(start2, end2, 10)

    x_single_x = pd.DataFrame({"0": np.concatenate([vals1, vals2])})
    x_single = x_single_x.copy()
    x_single["1"] = selected_sex_id
    x_single=x_single.to_numpy()
    
    z_single=z_single.to_numpy()
    
    x_single = inscaler.transform(x_single)
    be_single = np.full(x_single.shape[0], 3, dtype=float)
    q_single = model.get_mcmc_quantiles(X = x_single,batch_effects=be_single, z_scores=z_single) - sigma_intercept_mu * offsets[model_be]
    q_single = outscaler.inverse_transform(q_single)
    
    X_train = np.stack((data["age"], np.full(data.shape[0], selected_sex_id)),axis = 1)
    X_train_x = inscaler.transform(X_train)
    #X_train_x = X_train_x[:,0]
    be_single_x = np.full(data.shape[0], 3, dtype=float)
    Y_train2 = data[0].to_numpy().reshape(-1, 1)
    q_single_x = model.get_mcmc_quantiles(X = X_train_x,batch_effects=be_single_x, z_scores=Y_train2) - sigma_intercept_mu * offsets[model_be]
    q_single_x = outscaler.inverse_transform(q_single_x)

    #plt.xlim(np.min(x),np.max(x))
    plt.xlim(65,78)
    plt.ylim(10000,35000)
    for ir, r in enumerate(ran):
         if r == 0:
             plt.plot(x[:,0], q[:,ir], color = 'black')
         elif abs(r) == 3:
             plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linestyle="--", linewidth = 1)

         else:
             plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linewidth = 1)
    lmap = {'blr':'W-BLR','SHASHo':'$\mathcal{S}_o$', 'SHASHb_1':'$\mathcal{S}_{b1}$','SHASHb_2':'$\mathcal{S}_{b2}$',  'Normal':'$\mathcal{N}$'}

    q_single= q_single.flatten()
    x_single_x = x_single_x.to_numpy()
    from itertools import batched
    mint = "#98FF98"
    b_size = 10
    for (this_x, this_y) in zip(batched(x_single_x, b_size), batched(q_single, b_size)):
        plt.plot(this_x, this_y, linewidth=1.0, color=mint) # closes the shape automatically
    
  
    #polygon_x = [this_x[0], *this_x, this_x[-1]]
    #polygon_y = [0,        *this_y, 0]

    #plt.fill(polygon_x, polygon_y, color=mint, alpha=0.3)
    
    
    # Build color map
    diagnosis_codes = data["DIAGNOSIS"].dropna().unique()
    color_map = {diag: color for diag, color in zip(diagnosis_codes, plt.cm.tab10.colors)}
    color_map[np.nan] = "lightgrey"

    # Slice diagnosis values to match the 7 rows in X_train
    diagnosis_subset = data.loc[data.index[:len(X_train)], "DIAGNOSIS"]

    # Map diagnosis values to colors
    mapped_colors = diagnosis_subset.map(color_map).fillna("lightgrey").to_numpy()

    
    
    plt.plot(
        X_train[:, 0],
        q_single_x.squeeze(),
        alpha=0.7,         # transparency
        linewidth=2.5,     # line thickness
        color=cols[0],     # your chosen color
        )
    plt.scatter(
        X_train[:, 0],
        q_single_x.squeeze(),
        #c=data["DIAGNOSIS"].map(diagnosis_to_color),  # map DIAGNOSIS → color
        c=mapped_colors,
        edgecolor="black",
        zorder=5,
        s=60
        )
    # for area in areas:
    #     n,a,b,c,d,e,f = area
    #     rect = patches.Rectangle((a,b),c,d,label=n, linewidth=1,edgecolor = 'red',facecolor='None',zorder=10)
    #     plt.gca().add_patch(rect)
    #     plt.text(a+e*c, b+f*d, n, color='red',fontsize=16)
    # if len(areas) > 0:
    #     suffix = '_ann'
    # else:
    #     suffix = ''
    suffix = "fit"
    plt.title(lmap[l] + " " + feature, fontsize = 16)
    plt.xlabel('Age',fontsize=15)
#     plt.ylabel(feature,fontsize=12)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(linestyle=":", linewidth=1, alpha=0.7)
    fig.axes[0].yaxis.offsetText.set_fontsize(14)
    #plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/{l}_estimate_scaled_fixed_{atlas}_demented_adults_ADNI/batch_{f_idx+1}/mcmc_quantile_plot_{feature}_{l}_{suffix}_estimate_scaled_fixed_{atlas}_demented_adults.png",bbox_inches='tight',dpi=300)
    #plt.savefig(f"{l}_estimate_scaled_fixed_{atlas}_retrain/batch_{f_idx+1}/mcmc_quantile_plot_{feature}_{l}_{suffix}_retrain_{atlas}_adults.png",bbox_inches='tight',dpi=300)
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
    
    # except:
    #    print("not found")

#%%
#%%
def plot_MAP_quantiles_Figure(feature ="L_bankssts", l='SHASHo',selected_sex='female', atlas ='SC', data = data):
    f_idx = features.index(feature)

    #try:
    model_path = f"{l}_estimate_scaled_fixed_{atlas}_retrain_ADNI/batch_{f_idx+1}/Models/NM_0_0_estimate.pkl"
    model = pd.read_pickle(model_path)
    print(model.configs)


    inscaler = scaler("standardize")
    outscaler = scaler("standardize")
    selected_sex_id = 0 if selected_sex == 'female' else 1
    this_Y_train = Y_train_large[f_idx].to_numpy()
    #this_Y_train = Y_train[f_idx]
    this_scaled_X_train = inscaler.fit_transform(X_train_large)
    this_scaled_Y_train = outscaler.fit_transform(this_Y_train)
    
    #this_Y_test = Y_test[f_idx].to_numpy()
    #this_Y_test = Y_test[f_idx]
    #this_scaled_X_test = inscaler.transform(X_test)
    #this_scaled_Y_test = outscaler.transform(this_Y_test)

    train_sex_idx = np.where(X_train_large[:,1]==selected_sex_id)
    
    #test_sex_idx = np.where(X_test[:,1]==selected_sex_id)

    # select a model batch effect (69,1)

    model_be = [3]
    print(model_be)
    mu_intercept_mu = model.hbr.idata.posterior['mu_intercept_mu'].to_numpy().mean()
    sigma_intercept_mu = model.hbr.idata.posterior['sigma_intercept_mu'].to_numpy().mean()
    offsets = model.hbr.idata.posterior['offset_intercept_mu'].to_numpy().mean(axis = (0,1))
    model_offset_intercept_mu_be = offsets[model_be]


    # Make an empty array
    centered_Y_train = np.zeros_like(this_Y_train)
    #centered_Y_test = np.zeros_like(this_Y_test)

    # For each batch effect
    for i, be in enumerate(np.unique(Z_train_large)):
        this_offset_intercept = offsets[i]
        idx = (Z_train_large == be).all(1)

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
    #plt.scatter(X_train[:,0], Y_train[:,None], alpha = 0.1, s = 12, color=cols[0])
   
    #plt.scatter(X_test[test_sex_idx,0], Y_test[test_sex_idx,None], alpha = 0.1, s = 12, color=cols[1])
   
    # all sites
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[train_sex_idx,0], outscaler.inverse_transform(centered_Y_train[train_sex_idx,None]), alpha = 0.1, s = 12, color=cols[0])
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[test_sex_idx,0], outscaler.inverse_transform(centered_Y_test[test_sex_idx,None]), alpha = 0.1, s = 12, color=cols[1])
   
    
    be_map =np.unique(Z_train_large)
    print(be_map[model_be])
    idx = (Z_train_large == be_map[model_be[0]]).all(1)
    #plt.scatter(X_train[idx,0], centered_Y_train[idx,None], alpha = 0.1, s = 12, color=cols[0])
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[idx,0], outscaler.inverse_transform(centered_Y_train[idx,None]), alpha = 0.1, s = 12, color=cols[0])
  
    #idx = (Z_test == be_map[model_be[0]]).all(1)
    #plt.scatter(X_test[idx,0], centered_Y_test[idx,None], alpha = 0.1, s = 12, color=cols[1])
    #plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[idx,0], outscaler.inverse_transform(centered_Y_test[idx,None]), alpha = 0.1, s = 12, color=cols[1])
   

    difX = np.max(this_scaled_X_train[:,0])-np.min(X_train_large[:,0])
    #min0 = np.min(this_scaled_X_train[:,0]) + 0.01*difX
    #max0 = np.max(this_scaled_X_train[:,0]) - 0.01*difX
    min0 = np.min(this_scaled_X_train[:,0]) + 0.1*difX
    max0 = np.max(this_scaled_X_train[:,0]) - 0.1*difX
    sex = np.unique(this_scaled_X_train[:,1])[selected_sex_id]
    synthetic_X0 = np.linspace(min0, max0, 100)[:,None]
    #plt.xlim(min0,max0)
    synthetic_X = np.concatenate((synthetic_X0, np.full(synthetic_X0.shape,sex)),axis = 1)
    ran = np.arange(-3,4)

    # q = get_single_quantiles(synthetic_X,ran, model, model_be,MAP)-sigma_intercept_mu*offsets[tuple(model_be)]
    model_be_long = np.repeat(np.array(be_map[model_be]),synthetic_X.shape[0])
    #q = model.get_mcmc_quantiles(synthetic_X, model_be_long, z_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]))  - sigma_intercept_mu * offsets[model_be]
    q = model.get_mcmc_quantiles(synthetic_X, model_be_long)  - sigma_intercept_mu * offsets[model_be]
    q = outscaler.inverse_transform(q).T

    x = inscaler.inverse_transform(synthetic_X)
    
    
    z_single = data[0].to_numpy().reshape(-1, 1)
           

    x_single = np.stack((data["age"], np.full(data.shape[0], selected_sex_id)),axis = 1)
    

    #x_single=x_single.to_numpy()
    

    
    x_single = inscaler.transform(x_single)
    be_single = np.full(x_single.shape[0], 3, dtype=float)
    q_single = model.get_mcmc_quantiles(X = x_single,batch_effects=be_single, z_scores=z_single) - sigma_intercept_mu * offsets[model_be]
    q_single = outscaler.inverse_transform(q_single)
    
    #x_single = inscaler.transform(x_single)
    #X_train_x = X_train_x[:,0]
    
    
    #plt.xlim(np.min(x),np.max(x))
    plt.xlim(78,92)
    plt.ylim(15000,35000)
    for ir, r in enumerate(ran):
         if r == 0:
             plt.plot(x[:,0], q[:,ir], color = 'black')
         elif abs(r) == 3:
             plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linestyle="--", linewidth = 1)

         else:
             plt.plot(x[:,0], q[:,ir], color = 'black', alpha =0.6, linewidth = 1)
    lmap = {'blr':'W-BLR','SHASHo':'$\mathcal{S}_o$', 'SHASHb_1':'$\mathcal{S}_{b1}$','SHASHb_2':'$\mathcal{S}_{b2}$',  'Normal':'$\mathcal{N}$'}

    q_single= q_single.flatten()
    #x_single = x_single.to_numpy()
    
    
    diagnosis_codes = np.sort(data["DIAGNOSIS"].dropna().unique())

       # 2. Assign tab10 colors dynamically
    tab10 = plt.cm.tab10.colors
    color_map = {diag: tab10[i % len(tab10)] for i, diag in enumerate(diagnosis_codes)}

    # Optional: map NaN explicitly to grey
    color_map[np.nan] = "lightgrey"

       # 3. Plot
    groups = data.groupby("ID_subject").groups
    q = np.asarray(q_single) 
    
    for subject_id, idx in groups.items():
        x = data.loc[idx, "age"].to_numpy()
        y = q[idx]

        diagnoses = data.loc[idx, "DIAGNOSIS"].to_numpy()

        # Draw grey line
        plt.plot(x, y, linewidth=2.0, color="grey", alpha=0.8)

       # Overlay colored points
        point_colors = [color_map.get(d, "black") for d in diagnoses]
        plt.scatter(x, y, c=point_colors, edgecolor="black", zorder=5, s=40)
                    
        last_age = x[-1]
        last_score = y[-1]
        # plt.text(
        #     last_age + 0.5,        # shift to the right so it doesn’t overlap the marker
        #     last_score,
        #     str(subject_id),       # label = subject ID
        #     fontsize=8,
        #     color="black",
        #     alpha=0.7
        #     )
 
    

    suffix = "fit"
    plt.title(lmap[l] + " " + feature, fontsize = 16)
    plt.xlabel('Age',fontsize=15)
#     plt.ylabel(feature,fontsize=12)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(linestyle=":", linewidth=1, alpha=0.7)
    fig.axes[0].yaxis.offsetText.set_fontsize(14)
    plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE/all_subjects_4_normal_space_full.png",bbox_inches='tight',dpi=300)
    #plt.savefig(f"{l}_estimate_scaled_fixed_{atlas}_retrain/batch_{f_idx+1}/mcmc_quantile_plot_{feature}_{l}_{suffix}_retrain_{atlas}_adults.png",bbox_inches='tight',dpi=300)
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
    
    # except:
    #    print("not found")
#%%
features = SC_idp_cols()
features = DK_idp_cols()
features = DES_idp_cols()
#%%
X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_retrain_DK_adults.pkl",'rb')).to_numpy()
Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_retrain_DK_adults.pkl",'rb'))
Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_retrain_DK_adults.pkl",'rb')).to_numpy()

X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_retrain_DK_adults.pkl",'rb')).to_numpy()
Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_retrain_DK_adults.pkl",'rb'))
Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_retrain_DK_adults.pkl",'rb')).to_numpy()


#%%
X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_SC_demented_adults_ADNI.pkl",'rb')).to_numpy()
Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_SC_demented_adults_ADNI.pkl",'rb'))
Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_SC_demented_adults_ADNI.pkl",'rb')).to_numpy()
X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_test_SC_demented_adults_ADNI.pkl",'rb')).to_numpy()
Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_test_SC_demented_adults_ADNI.pkl",'rb'))
Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/tebefile_SC_demented_adults_ADNI.pkl",'rb')).to_numpy()


# In[5]:
for i, feat in enumerate(features):
    if  i == 1:
        break
    plot_MAP_quantiles(l='SHASHb_1', feature=feat, atlas ='SC')


#%%
for i, feat in enumerate(features):
    if  i == 1:
        break
    plot_MAP_quantiles_Figure(l='SHASHb_1', feature=feat, atlas ='SC', data=data)

