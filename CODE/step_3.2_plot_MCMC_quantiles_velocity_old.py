
# coding: utf-8

# In[1]:


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
from utils import return_idp_cols


# In[2]:


features = return_idp_cols()

selected_feature = features[0]


# In[3]:


# Get data
X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_train_cross.pkl",'rb')).to_numpy()
Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_train_cross.pkl",'rb'))
Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/trbefile_cross.pkl",'rb')).to_numpy()
X_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/X_test_cross.pkl",'rb')).to_numpy()
Y_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/Y_test_cross.pkl",'rb'))
Z_test = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data/tebefile_cross.pkl",'rb')).to_numpy()
Z_train.shape
X_train.shape

#%%

Z_train.groupby(by=["b"]).sum() # try to find the lagest stite

# In[4]:
os.chdir('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Data')
processing_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/'

def plot_MAP_quantiles(fold = 0, feature ="L_bankssts", l='Normal',selected_sex='female', areas = []):
    f_idx = features.index(feature)
    
    # map_path  = f'10_folds_results/MAPS/MAP_fold{fold}_{f_idx}_{l}.pkl'
#     map_path = f"compare_linear_epsilon_and_delta_wdir/MAP_lifespan_{feature}_SHASHb_{postfix1}_{postfix2}.pkl"
    # with open(map_path,'rb') as file:
    #     MAP = pickle.load(file)
    model_path = f'cross_{fold}_{l}_fit_unscaled_fixed/batch_{f_idx+1}/Models/NM_0_0_fit.pkl'
    with open(os.path.join(processing_dir, model_path),'rb') as file:
        model = pickle.load(file)
    
    this_Y_train = Y_train[feature_i].to_numpy()
    this_Y_test = Y_test[feature_i].to_numpy()
    #inscaler = scaler("standardize")
    #outscaler = scaler("standardize")
    #this_scaled_X_train = inscaler.fit_transform(X_train)
    #this_scaled_Y_train = outscaler.fit_transform(this_Y_train)
    #this_scaled_X_test = inscaler.transform(X_test)
    #this_scaled_Y_test = outscaler.transform(this_Y_test)
    
    selected_sex_id = 0 if selected_sex == 'female' else 1
    train_sex_idx = np.where(X_train[:,1]==selected_sex_id)
    test_sex_idx = np.where(X_test[:,1]==selected_sex_id)

    # select a model batch effect (69,1)
    model_be = [0]
    mu_intercept_mu = model.hbr.idata.posterior['mu_intercept_mu'].to_numpy().mean()
    sigma_intercept_mu = model.hbr.idata.posterior['sigma_intercept_mu'].to_numpy().mean()
    offsets = model.hbr.idata.posterior['offset_intercept_mu'].to_numpy().mean(axis = (0,1))
    model_offset_intercept_mu_be = offsets[model_be]

    
    # Make an empty array
    centered_Y_train = np.zeros_like(this_scaled_Y_train)
    centered_Y_test = np.zeros_like(this_scaled_Y_test)
    
    # For each batch effect
    for be in np.unique(Z_train,axis = 0):
        this_offset_intercept = offsets[be]
        idx = (Z_train == be).all(1) 

        centered_Y_train[idx] = this_scaled_Y_train[idx]-sigma_intercept_mu*this_offset_intercept
        idx = (Z_test == be).all(1) 
        centered_Y_test[idx] = this_scaled_Y_test[idx]-sigma_intercept_mu*this_offset_intercept

    fig = plt.figure(figsize=(5,4))

    ytrain_inv = outscaler.inverse_transform(centered_Y_train[train_sex_idx,None])
    maxy = np.max(ytrain_inv)
    miny = np.min(ytrain_inv)
    dify = maxy - miny
    plt.ylim(miny - 0.1*dify, maxy + 0.1*dify )
    plt.scatter(inscaler.inverse_transform(this_scaled_X_train)[train_sex_idx,0], outscaler.inverse_transform(centered_Y_train[train_sex_idx,None]), alpha = 0.1, s = 12, color=cols[0])
    plt.scatter(inscaler.inverse_transform(this_scaled_X_test)[test_sex_idx,0], outscaler.inverse_transform(centered_Y_test[test_sex_idx,None]), alpha = 0.1, s = 12, color=cols[1])
    
    difX = np.max(this_scaled_X_train[:,0])-np.min(this_scaled_X_train[:,0])
    min0 = np.min(this_scaled_X_train[:,0]) + 0.01*difX
    max0 = np.max(this_scaled_X_train[:,0]) - 0.01*difX
    sex = np.unique(this_scaled_X_train[:,1])[selected_sex_id]
    synthetic_X0 = np.linspace(min0, max0, 200)[:,None]
#     plt.xlim(min0,max0)
    synthetic_X = np.concatenate((synthetic_X0, np.full(synthetic_X0.shape,sex)),axis = 1)
    ran = np.arange(-3,4)
    
    # q = get_single_quantiles(synthetic_X,ran, model, model_be,MAP)-sigma_intercept_mu*offsets[tuple(model_be)]
    model_be = np.repeat(np.array(model_be),synthetic_X.shape[0])
    q = model.get_mcmc_quantiles(synthetic_X, model_be) - sigma_intercept_mu * offsets[model_be[0]]
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
    
    for area in areas:
        n,a,b,c,d,e,f = area
        rect = patches.Rectangle((a,b),c,d,label=n, linewidth=1,edgecolor = 'red',facecolor='None',zorder=10)
        plt.gca().add_patch(rect)
        plt.text(a+e*c, b+f*d, n, color='red',fontsize=16)
    if len(areas) > 0:
        suffix = '_ann'
    else:
        suffix = ''
        
    plt.title(lmap[l], fontsize = 16)
    plt.xlabel('Age',fontsize=15)
#     plt.ylabel(feature,fontsize=12)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(linestyle=":", linewidth=1, alpha=0.7)
    fig.axes[0].yaxis.offsetText.set_fontsize(14)
#     plt.savefig(f"/home/preclineu/stijdboe/Projects/MasterThesis/Latex/Thesis/imgs/mcmc_quantile_plot_{feature}_{l}{suffix}.png",bbox_inches='tight',dpi=300)
    plt.show()
    del model 
    del this_Y_train
    del this_Y_test
    del this_offset_intercept
    del this_scaled_X_test
    del this_scaled_X_train
    del this_scaled_Y_test
    del this_scaled_Y_train
    gc.collect()
    


# In[5]:


plot_MAP_quantiles(l='Normal', feature='Right-Lateral-Ventricle', areas = [('A',3,0,14,3e4,0.1, 0.8), ('B',70,-5000,28,30000, 0.1, 0.1)])
plot_MAP_quantiles(l='SHASHo', feature='Right-Lateral-Ventricle', areas = [('A',3,0,14,3e4,0.1, 0.8), ('B',70,-5000,28,30000,0.1, 0.1)])
plot_MAP_quantiles(l='SHASHb_1', feature='Right-Lateral-Ventricle', areas = [('A',3,0,14,3e4,0.1, 0.8), ('B',70,-5000,28,30000,0.1, 0.1)])
plot_MAP_quantiles(l='SHASHb_2', feature='Right-Lateral-Ventricle', areas = [('A',3,0,14,3e4,0.1, 0.8), ('B',70,-5000,28,30000,0.1, 0.1)])


# plot_MAP_quantiles(l='SHASHb_2', feature='Right-Lateral-Ventricle', areas = [('A',3,0,14,3e4,0.1, 0.8), ('B',70,-5000,28,30000,0.1, 0.1)])


# In[6]:


#plot_MAP_quantiles(l='Normal', feature='WM-hypointensities', areas = [('C',3,-1e4,14,3e4,0.6, 0.8)])
plot_MAP_quantiles(l='SHASHo', feature='WM-hypointensities', areas = [('C',3,-1e4,14,3e4,0.6, 0.8)])
plot_MAP_quantiles(l='SHASHb_1', feature='WM-hypointensities', areas = [('C',3,-2e3,14,3e4,0.6, 0.8)])
plot_MAP_quantiles(l='SHASHb_2', feature='WM-hypointensities', areas = [('C',3,-2e3,14,3e4,0.6, 0.8)])


# In[8]:




for i_f, f in enumerate(features):
    plot_MAP_quantiles(l='Normal', feature=f)
    plot_MAP_quantiles(l='SHASHo', feature=f)
    plot_MAP_quantiles(l='SHASHb_1', feature=f)
    plot_MAP_quantiles(l='SHASHb_2', feature=f)

# plot_MAP_quantiles(postfix2='True',feature=selected_feature, selected_sex = 'female')


# In[8]:


plot_MAP_quantiles(postfix2='True', selected_sex = 'male')
plot_MAP_quantiles(postfix2='True', selected_sex = 'female')


# In[11]:


plot_MAP_quantiles(l = 'SHASHb',feature=  'Right-Lateral-Ventricle')


# In[10]:


dim1 = 10

all_bes = np.unique(np.random.randint(0,dim1,size=(10,2)), axis = 0)

az = np.random.randn(dim1, dim1)
for be in all_bes:
    print(be)
    
    bet = tuple(be)
    print(az[bet])


# In[35]:


import pylab as pl
import numpy as np

a = np.array([[0,0.5]])
pl.figure(figsize=(0.2, 10))
img = pl.imshow(a, cmap="viridis")
plt.gca().set_visible(False)
cax = pl.axes([0.1, 0.2, 0.8, 0.6])
pl.colorbar(orientation="vertical", cax=cax)
pl.savefig(f'/home/preclineu/stijdboe/Projects/MasterThesis/Latex/Thesis/imgs/AUC_images/colorbar.pdf',bbox_inches='tight')

