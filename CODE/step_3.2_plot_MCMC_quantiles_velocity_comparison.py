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
from utils import return_idp_cols, DES_idp_cols, sc_idp_cols
from matplotlib.backends.backend_pdf import PdfPages

#%%

# cortical 
#features = return_idp_cols()
#features = DES_idp_cols()

#selected_feature = features[0]

#subcortical idps
features = sc_idp_cols()

# In[3]:


# # Get data
# X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_train_delta_fixed_DK.pkl",'rb')).to_numpy()
# Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_train_delta_fixed_L_bankssts_DK.pkl",'rb'))
# Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/trbefile_delta_fixed_DK.pkl",'rb')).to_numpy()
# X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_test_delta_fixed_DK.pkl",'rb')).to_numpy()
# Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_test_delta_fixed_DK_L_bankssts_DK.pkl",'rb'))
# Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/tebefile_delta_fixed_DK.pkl",'rb')).to_numpy()
# Z_train.shape
# X_train.shape

# Get data


#X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_train_cross_DK.pkl",'rb')).to_numpy()
#Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_train_cross_DK.pkl",'rb'))
#Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/trbefile_cross_DK.pkl",'rb')).to_numpy()
#X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_test_cross_DK.pkl",'rb')).to_numpy()
#Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_test_cross_DK.pkl",'rb'))
#Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/tebefile_cross_DK.pkl",'rb')).to_numpy()
#Z_train.shape
#X_train.shape

#%% cortical
# features = return_idp_cols()
X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_train_cross_sc.pkl",'rb')).to_numpy()
Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_train_cross_sc.pkl",'rb')).to_numpy()
Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/trbefile_cross_sc.pkl",'rb')).to_numpy()
X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_test_cross_sc.pkl",'rb')).to_numpy()
Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_test_cross_sc.pkl",'rb')).to_numpy()
Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/tebefile_cross_sc.pkl",'rb')).to_numpy()

#%% subcortical measures
features = sc_idp_cols()
train = "cross"
test = "long"
X_train = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_train_{train}_sc.pkl",'rb')).to_numpy()
Y_train = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_train_{train}_sc.pkl",'rb'))
Z_train = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/trbefile_{train}_sc.pkl",'rb')).to_numpy()
X_test = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_test_{test}_sc.pkl",'rb')).to_numpy()
Y_test = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_test_{test}_sc.pkl",'rb'))
Z_test = pickle.load(open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/tebefile_{test}_sc.pkl",'rb')).to_numpy()


# In[4]:
os.chdir('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity')
#model_path = f'fold_{fold}_{l}_scaled_estimate/Models/NM_0_0_estimatedeltaNormal.pkl'
#model_path = 'cross_0_SHASHo_fit_scaled_fixed_DK/Models/batch_1/Models/NM_0_0_fit.pkl'
#%%
def plot_MAP_quantiles(train = "cross", test = "long", model_be = 25, fold = 0, feature ="L_bankssts", l='SHASHo',selected_sex='female'):
    f_idx = features.index(feature)

    # map_path  = f'10_folds_results/MAPS/MAP_fold{fold}_{f_idx}_{l}.pkl'
#     map_path = f"compare_linear_epsilon_and_delta_wdir/MAP_lifespan_{feature}_SHASHb_{postfix1}_{postfix2}.pkl"
    # with open(map_path,'rb') as file:
    #     MAP = pickle.load(file)
    try:
        #model_path = f'cross_{fold}_{l}_fit_unscaled_fixed_DK/batch_{f_idx+1}/Models/NM_0_0_fit.pkl'
        #model_path = f'fold_{fold}_{l}_scaled_estimate/Models/NM_0_0_estimatedeltaNormal.pkl'
        #model_path = f'cross_{fold}_{l}_fit_scaled_fixed_DK/batch_{f_idx+1}/Models/NM_0_0_fit.pkl'
        # cortical 
        #model_path = f'{fold}_{l}_fit_scaled_fixed_DK_CT/batch_{f_idx+1}/Models/NM_0_0_fit.pkl'
        # subcortical
        model_path = f'train-{train}_test-{test}_estimate_scaled_fixed_sc_{l}/batch_{f_idx+1}/Models/NM_0_0_estimate.pkl'
        #model_path = model_path
        #with open(model_path,'rb') as file:
            #model = pickle.load(file)
        model = pd.read_pickle(model_path)
    except:
        print("not found")

    this_Y_train = Y_train[f_idx].to_numpy()
    this_Y_test = Y_test[f_idx].to_numpy()
    inscaler = scaler("standardize")
    outscaler = scaler("standardize")
    this_scaled_X_train = inscaler.fit_transform(X_train)
    this_scaled_Y_train = outscaler.fit_transform(this_Y_train)
    this_scaled_X_test = inscaler.transform(X_test)
    this_scaled_Y_test = outscaler.transform(this_Y_test)

    selected_sex_id = 0 if selected_sex == 'female' else 1
    train_sex_idx = np.where(X_train[:,1]==selected_sex_id)
    test_sex_idx = np.where(X_test[:,1]==selected_sex_id)

    # select a model batch effect (69,1)
    #model_be = [25]
    model_be = model_be
    mu_intercept_mu = model.hbr.idata.posterior['mu_intercept_mu'].to_numpy().mean()
    sigma_intercept_mu = model.hbr.idata.posterior['sigma_intercept_mu'].to_numpy().mean()
    offsets = model.hbr.idata.posterior['offset_intercept_mu'].to_numpy().mean(axis = (0,1))
    #model_offset_intercept_mu_be = offsets[model_be]


    # Make an empty array
    centered_Y_train = np.zeros_like(this_Y_train)
    centered_Y_test = np.zeros_like(this_Y_test)

    # For each batch effect
    for i, be in enumerate(np.unique(Z_train)):
        this_offset_intercept = offsets[i]
        idx = (Z_train == be).all(1)

        centered_Y_train[idx] = this_scaled_Y_train[idx]-sigma_intercept_mu*this_offset_intercept
        #centered_Y_train[idx] = this_Y_train[idx]
        idx = (Z_test == be).all(1)
        centered_Y_test[idx] = this_scaled_Y_test[idx]-sigma_intercept_mu*this_offset_intercept
        #centered_Y_test[idx] = this_Y_test[idx]
    fig = plt.figure(figsize=(5,4))

    ytrain_inv = outscaler.inverse_transform(centered_Y_train[train_sex_idx,None])
    maxy = np.max(ytrain_inv)
    miny = np.min(ytrain_inv)
    dify = maxy - miny
    #plt.ylim(miny - 0.1*dify, maxy + 0.1*dify )
    #plt.scatter(X_train[train_sex_idx,0], Y_train[train_sex_idx,None], alpha = 0.1, s = 12, color=cols[0])
    #plt.scatter(X_test[test_sex_idx,0], Y_test[test_sex_idx,None], alpha = 0.1, s = 12, color=cols[1])
   
    plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[train_sex_idx,0], outscaler.inverse_transform(centered_Y_train[train_sex_idx,None]), alpha = 0.1, s = 12, color=cols[0])
    plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[test_sex_idx,0], outscaler.inverse_transform(centered_Y_test[test_sex_idx,None]), alpha = 0.1, s = 12, color=cols[1])
   
    
    be_map =np.unique(Z_train)
    #idx = (Z_train == be_map[model_be[0]]).all(1)
   # plt.scatter(X_train[idx,0], centered_Y_train[idx,None], alpha = 0.1, s = 12, color=cols[0])
    i#dx = (Z_test == be_map[model_be[0]]).all(1)
    #plt.scatter(X_test[idx,0], centered_Y_test[idx,None], alpha = 0.1, s = 12, color=cols[1])

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

    # for area in areas:
    #     n,a,b,c,d,e,f = area
    #     rect = patches.Rectangle((a,b),c,d,label=n, linewidth=1,edgecolor = 'red',facecolor='None',zorder=10)
    #     plt.gca().add_patch(rect)
    #     plt.text(a+e*c, b+f*d, n, color='red',fontsize=16)
    # if len(areas) > 0:
    #     suffix = '_ann'
    # else:
    #     suffix = ''
    suffix = "estimate"
    plt.title(lmap[l] + " " + feature + str(be_map[model_be]), fontsize = 16)
    plt.xlabel('Age',fontsize=15)
#     plt.ylabel(feature,fontsize=12)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(linestyle=":", linewidth=1, alpha=0.7)
    fig.axes[0].yaxis.offsetText.set_fontsize(14)
    #plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/comparison/mcmc_quantile_plot_train-{train}_test-{test}_{feature}_{l}_{suffix}.png",bbox_inches='tight',dpi=300)
    plt.show()
    saved_figures.append(fig)  
    del model
    del this_Y_train
    del this_Y_test
    del this_offset_intercept
    del this_scaled_X_test
    del this_scaled_X_train
    del this_scaled_Y_test
    del this_scaled_Y_train
    gc.collect()
    # except Exception(e):
    #     print("not found:" + e)


# In[5]:
saved_figures = []
#models = ["Normal", "SHASHo","SHASHb_1"]
#models = ["Normal", "SHASHo"]
models =  ["SHASHo"]
for i, feat in enumerate(features):
    if i == 7:
        for m, mod in enumerate(models):
            for j in range(1):
                plot_MAP_quantiles(l=mod, feature=feat, model_be = j, train="long", test="long")
    #if  i == 8:
        #break

#%%
chunk_size =  25  # Number of plots per PDF
num_pdfs = len(saved_figures) // chunk_size

last_chunk = saved_figures[num_pdfs*chunk_size:]
#if last_chunk:
#    save_plots_to_pdf(f'plots_batch_{num_pdfs+1}.pdf', last_chunk)

for pdf_index in range(num_pdfs):
    pdf_name = f'plots_batch_{pdf_index + 1}.pdf'  # Name for each PDF
    with PdfPages(pdf_name) as pdf:
        # Get the current chunk of figures
        chunk = saved_figures[pdf_index * chunk_size:(pdf_index + 1) * chunk_size]
        for fig in chunk:
            pdf.savefig(fig)  # Save the figure to the current PDF
            plt.close(fig) 

#%%

# plot_MAP_quantiles(l='SHASHo', feature=selected_feature)
# #%%


# for i_f, f in enumerate(features):
#     plot_MAP_quantiles(l='Normal', feature=f)
#     plot_MAP_quantiles(l='SHASHo', feature=f)
#     plot_MAP_quantiles(l='SHASHb_1', feature=f)
#     plot_MAP_quantiles(l='SHASHb_2', feature=f)

# # plot_MAP_quantiles(postfix2='True',feature=selected_feature, selected_sex = 'female')


# # In[8]:


# plot_MAP_quantiles(postfix2='True', selected_sex = 'male')
# plot_MAP_quantiles(postfix2='True', selected_sex = 'female')


# # In[11]:


# plot_MAP_quantiles(l = 'SHASHb',feature=  'Right-Lateral-Ventricle')


# # In[10]:


# dim1 = 10

# all_bes = np.unique(np.random.randint(0,dim1,size=(10,2)), axis = 0)

# az = np.random.randn(dim1, dim1)
# for be in all_bes:
#     print(be)

#     bet = tuple(be)
#     print(az[bet])


# # In[35]:


# import pylab as pl
# import numpy as np

# a = np.array([[0,0.5]])
# pl.figure(figsize=(0.2, 10))
# img = pl.imshow(a, cmap="viridis")
# plt.gca().set_visible(False)
# cax = pl.axes([0.1, 0.2, 0.8, 0.6])
# pl.colorbar(orientation="vertical", cax=cax)
# pl.savefig(f'/home/preclineu/stijdboe/Projects/MasterThesis/Latex/Thesis/imgs/AUC_images/colorbar.pdf',bbox_inches='tight')

