#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:16:02 2025

@author: johbay
"""

import os
import pickle as pkl
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from utils import SC_idp_cols, DK_idp_cols, DES_idp_cols
import pandas as pd

features = SC_idp_cols()
#wdir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_demented_adults_ADNI/'
wdir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_SC_all_regions/results/Velocity/'

# fnam = os.path.join(wdir, 'batch_1/all_gains_Left-Lateral-Ventricle.pkl')
# with open(fnam,'rb') as f:
#     z = pkl.load(f)
#     inv_pred = False

# fnam = os.path.join(wdir, 'batch_1/cross_sectional.pkl')
# # 1 Hc, 2 EMC and 3 AD
# with open(fnam,'rb') as f:
#     zc = pkl.load(f)

# fnam  = os.path.join(wdir, 'batch_11/all_gains_Left-Hippocampus.pkl')
# with open(fnam,'rb') as f:
#     z = pkl.load(f)
#     inv_pred = True

non_invert_regions = [
    "Left-Lateral-Ventricle",
    "Right-Lateral-Ventricle",
    "3rd-Ventricle",
    "4th-Ventricle",
    "5th-Ventricle",
    "Left-Inf-Lat-Vent",
    "Right-Inf-Lat-Vent"
]

results = []

for i, feat in enumerate(features):
    fnam = os.path.join(wdir, f'{feat}/cross_sectional.pkl')
    with open(fnam,'rb') as f:
        zc = pkl.load(f)
        if feat in non_invert_regions:
            inv_pred = False
        else:
            inv_pred = True
    
    zc['diagnosis_transition'] = 'na'
    zc['diagnosis_transition'].loc[(zc['diagnosis_t1'] == 2.0) & (zc['diagnosis_t2'] == 3.0) ] = '(E)MCI → AD' 
    
    z = zc 
    
    # calculate the fpr and tpr for all thresholds of the classification
    probs = z['z_gain']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = z['diagnosis_transition'] == '(E)MCI → AD'
    if inv_pred:
        y_test = 1-y_test.astype(int)
    else:
        y_test = y_test.astype(int)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc_gain = metrics.auc(fpr, tpr)
    print('AUC gain: %0.2f' % auc_gain)
    
    plt.title(f'ROC - {feat} - z_gain')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_gain)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    probs = zc['Z1']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = zc['diagnosis_transition'] == '(E)MCI → AD'
    if inv_pred:
        y_test = 1-y_test.astype(int)
    else:
        y_test = y_test.astype(int)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc_z1 = metrics.auc(fpr, tpr)
    
    print('AUC cross-sectional 1: %0.2f' % auc_z1)
    
    plt.title(f'ROC - {feat} - Z1')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_z1)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    probs = zc['Z2']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = zc['diagnosis_transition'] == '(E)MCI → AD'
    if inv_pred:
        y_test = 1-y_test.astype(int)
    else:
        y_test = y_test.astype(int)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc_z2 = metrics.auc(fpr, tpr)
    print('AUC cross-sectional 2: %0.2f' % auc_z2)
    plt.title(f'ROC - {feat}  -Z2')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_z2)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    results.append({
        "feature": feat,
        "AUC_z_gain": auc_gain,
        "AUC_Z1": auc_z1,
        "AUC_Z2": auc_z2,
        "invert_prediction": inv_pred
    })
    
auc_table = pd.DataFrame(results)
auc_table.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/auc_table_SC.csv", sep='\t')



#%%
import ggseg
import matplotlib as mpl
data = dict(zip(auc_table["feature"], auc_table["AUC_z_gain"]))

ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC (z-gain)",
    vminmax=[0.3,0.9]
)


data = dict(zip(auc_table["feature"], auc_table["AUC_Z1"]))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC Z1",
    vminmax=[0.3,0.9]
)

data = dict(zip(auc_table["feature"], auc_table["AUC_Z2"]))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC Z2",
    vminmax=[0.3,0.9]
)

#%%

import ggseg
import matplotlib as mpl

auc_table["feature_new"] = (
    auc_table["feature"]
    .str.replace("^L_", "", regex=True)
    .str.replace("^R_", "", regex=True)
    + auc_table["feature"].str[0].map({'L': '_left', 'R': '_right'})
)


data = dict(zip(auc_table["feature_new"], auc_table["AUC_z_gain"]))

ggseg.plot_dk(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC (z-gain)",
    vminmax=[0.3,0.9]
)


data = dict(zip(auc_table["feature_new"], auc_table["AUC_Z1"]))
ggseg.plot_dk(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC Z1",
    vminmax=[0.3,0.9]
)

data = dict(zip(auc_table["feature_new"], auc_table["AUC_Z2"]))


ggseg.plot_dk(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC Z2",
    vminmax=[0.3,0.9]
)


