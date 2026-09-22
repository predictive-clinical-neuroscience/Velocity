#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:16:02 2025

@author: johbay

This script calculates the AUC curves for z-gain, Z1 and Z2. It then also calculates whether this AUC is significantly different from 0.5, and then 
each of the models with each other. Last, it re-calculates the AUC p values, fdr corrected.
"""

import os
import pickle as pkl
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from utils import SC_idp_cols, DK_idp_cols, DES_idp_cols
import pandas as pd
from utilities_test import test_write_dataframes_tsv
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy import stats
from utilities_stats import auc_vs_chance_pval, delong_pval

#%%


#%%

atlas = "DES"

if atlas == "SC":
    features = SC_idp_cols()
    features.remove('Left-Thalamus')
    features.remove('Right-Thalamus')
elif atlas == "DES":
    features = DES_idp_cols()
else:
    features = DK_idp_cols()

#wdir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_demented_adults_ADNI/'
wdir = f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_{atlas}_all_regions/results/Velocity/'
#
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
    "Right-Inf-Lat-Vent",
    "CSF"
]

results = []

slopes = pd.read_csv(os.path.join(wdir, f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/slopes_{atlas}.csv'))
slopes['diagnosis_transition'] = 'na'
slopes["diagnosis_t1"] = slopes["diagnosis_t1"].map({1: "HC", 2: "(E)MCI", 3: "AD" })
slopes["diagnosis_t2"] = slopes["diagnosis_t2"].map({1: "HC", 2: "(E)MCI", 3: "AD" })
slopes["diagnosis_transition"] = slopes["diagnosis_t1"] + " → " + slopes["diagnosis_t2"]

for i, feat in enumerate(features):
       
    #fnam = os.path.join(wdir, f'{feat}/cross_sectional.pkl') # this  is the old file with cross-sectional data
    fnam = os.path.join(wdir, f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/revision/z_gains_ADNI/{atlas}/clinical_HC_gains_ADNI_{feat}_2026_revision.pkl')
    with open(fnam,'rb') as f:
        z = pkl.load(f)
        if feat in non_invert_regions:
            inv_pred = False
        else:
            inv_pred = True
    
    #remove for all gains
    #z["diagnosis_t1"] = z["diagnosis_t1"].map({1: "HC", 2: "(E)MCI", 3: "AD" })
   # z["diagnosis_t2"] = z["diagnosis_t2"].map({1: "HC", 2: "(E)MCI", 3: "AD" })

    
    z['diagnosis_transition'] = 'na'
    ##z['diagnosis_transition'].loc[(z['diagnosis_t1'] == 2.0) & (z['diagnosis_t2'] == 3.0) ] = '(E)MCI → AD' 
    z["diagnosis_transition"] = z["diagnosis_t1"] + " → " + z["diagnosis_t2"]
    
    
    z = z.merge(
    slopes[["ID_subject", "age1", "age2", "t1_index", "t2_index", f"slope_{feat}"]],
    on=["ID_subject", "age1", "age2", "t1_index", "t2_index"],
    how="inner",
    )
    
    #keep = ['(E)MCI → AD', '(E)MCI → (E)MCI']
    #z = z[z["diagnosis_transition"].isin(keep)]
    
    # calculate the for and tpr for all thresholds of the classification
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
    p_chance_gain = auc_vs_chance_pval(probs, y_test)   # <-- added
    probs_gain, y_gain = probs, y_test                  # <-- added (save for DeLong)

    
    plt.title(f'ROC - {feat} - z_gain')
    plt.plot(fpr, tpr, 'r', label = 'AUC Z-gain = %0.2f' % auc_gain)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    
    probs = z['Z1']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = z['diagnosis_transition'] == '(E)MCI → AD'
    if inv_pred:
        y_test = 1-y_test.astype(int)
    else:
        y_test = y_test.astype(int)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc_z1 = metrics.auc(fpr, tpr)
    
    print('AUC cross-sectional 1: %0.2f' % auc_z1)
    p_chance_z1 = auc_vs_chance_pval(probs, y_test)     # <-- added
    probs_z1 = probs 
    
    plt.title(f'ROC - {feat} - Z1')
    plt.plot(fpr, tpr, 'b', label = 'AUC Z1 = %0.2f' % auc_z1)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    
    probs = z['Z2']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = z['diagnosis_transition'] == '(E)MCI → AD'
    if inv_pred:
        y_test = 1-y_test.astype(int)
    else:
        y_test = y_test.astype(int)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc_z2 = metrics.auc(fpr, tpr)
    print('AUC cross-sectional 2: %0.2f' % auc_z2)
    
    p_chance_z2 = auc_vs_chance_pval(probs, y_test)     # <-- added
    probs_z2 = probs  
    
    plt.title(f'ROC - {feat}  -Z2')
    plt.plot(fpr, tpr, 'g', label = 'AUC Z2 = %0.2f' % auc_z2)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    

    
    probs = z[f'slope_{feat}']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = z['diagnosis_transition'] == '(E)MCI → AD'
    if inv_pred:
        y_test = 1-y_test.astype(int)
    else:
        y_test = y_test.astype(int)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc_slope = metrics.auc(fpr, tpr)
    print('AUC slope: %0.2f' % auc_slope)
    p_chance_slope = auc_vs_chance_pval(probs, y_test)   # <-- added
    probs_slope = probs                # <-- added (save for DeLong)

    
    #plt.title(f'ROC - {feat} - zlopes)
    plt.title(f'ROC - {feat}' )
    plt.plot(fpr, tpr, 'gray', label = 'AUC slope = %0.2f' % auc_gain)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
    p_gain_vs_z1 = delong_pval(y_gain, probs_gain, probs_z1)
    p_gain_vs_z2 = delong_pval(y_gain, probs_gain, probs_z2)
    p_z1_vs_z2   = delong_pval(y_gain, probs_z1,   probs_z2)
    p_gain_vs_slope = delong_pval(y_gain, probs_gain, probs_slope)

    
    results.append({
        "feature": feat,
        "AUC_z_gain": auc_gain,
        "AUC_Z1": auc_z1,
        "AUC_Z2": auc_z2,
        "AUC_slope": auc_slope,
        "invert_prediction": inv_pred,
        "p_chance_z_gain": p_chance_gain,   # <-- added
        "p_chance_Z1": p_chance_z1,         # <-- added
        "p_chance_Z2": p_chance_z2,
        "p_chance_slope": p_chance_slope,         # <-- added
        "p_delong_gain_vs_Z1": p_gain_vs_z1, # <-- added
        "p_delong_gain_vs_Z2": p_gain_vs_z2, # <-- added
        "p_delong_Z1_vs_Z2": p_z1_vs_z2, # <-- added
        "p_delong_gain_vs_slope": p_gain_vs_slope
    })
    
auc_table = pd.DataFrame(results)

dataframes = {
    "auc_table": auc_table,
}

#test_write_dataframes_tsv(dataframes, tags=[atlas], write_dir="/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/")


for col in [c for c in auc_table.columns if c.startswith(("p_chance", "p_delong"))]:
    mask = auc_table[col].notna()
    auc_table.loc[mask, col + "_fdr"] = multipletests(auc_table.loc[mask, col], method="fdr_bh")[1]
    
p_cols = [c for c in auc_table.columns if c.startswith(("p_chance", "p_delong"))]

auc_table_display = auc_table.copy()
auc_table_display[p_cols] = auc_table_display[p_cols].map(
    lambda p: "n.s." if pd.notna(p) and p > 0.05 else p
)

auc_table.to_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/auc_table_{atlas}_revision_reduced.csv", sep='\t')



#%%
import ggseg
import matplotlib as mpl
import numpy as np

atlas ="SC"
auc_table = pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/auc_table_{atlas}_revision_reduced_two_categories.csv", sep='\t')

auc_table["difference"] = auc_table["AUC_z_gain"] - auc_table["AUC_slope"]
#data = dict(zip(auc_table["feature"], auc_table["AUC_z_gain"]))
auc_masked = auc_table["AUC_z_gain"].where(auc_table["p_chance_z_gain_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature"], auc_masked))

ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC (z-gain)",
    vminmax=[0.5,0.8]
)


auc_masked = auc_table["AUC_Z1"].where(auc_table["p_chance_Z1_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature"], auc_masked))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (Z1)",
    title="Subcortical AUC Z1",
    vminmax=[0.5,0.8]
)

auc_masked = auc_table["AUC_Z2"].where(auc_table["p_chance_Z2_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature"], auc_masked))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (Z2)",
    title="Subcortical AUC Z2",
    vminmax=[0.5,0.8]
)

auc_masked = auc_table["AUC_slope"].where(auc_table["p_chance_slope_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature"], auc_masked))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (raw slopes)",
    title="Subcortical AUC raw slopes",
    vminmax=[0.5,0.8]
)

auc_masked = auc_table["difference"].where(auc_table["p_delong_gain_vs_slope"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature"], auc_masked))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("coolwarm"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC gain",
    title="Regions where gain model is better than slope model",
    vminmax=[-0.1,0.1]
)



#%%

import ggseg
import matplotlib as mpl

atlas = "DK"
auc_table = pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/auc_table_{atlas}_revision_reduced_two_categories.csv", sep='\t')

auc_table["feature_new"] = (
    auc_table["feature"]
    .str.replace("^L_", "", regex=True)
    .str.replace("^R_", "", regex=True)
    + auc_table["feature"].str[0].map({'L': '_left', 'R': '_right'})
)

auc_table["difference"] = auc_table["AUC_z_gain"] - auc_table["AUC_slope"]
#data = dict(zip(auc_table["feature"], auc_table["AUC_z_gain"]))

cmap = mpl.cm.get_cmap("Blues").copy()   # copy so you don't mutate the global cmap
cmap.set_bad("gray")

auc_masked = auc_table["AUC_z_gain"].where(auc_table["p_chance_z_gain_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))

#data = dict(zip(auc_table["feature_new"], auc_table["AUC_z_gain"]))

ggseg.plot_dk(
    data,
    #cmap=mpl.cm.get_cmap("Blues"), 
    cmap=cmap,         # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC (z-gain)",
    vminmax=[0.5,0.8]
)

# data = dict(zip(auc_table["feature_new"], auc_table["AUC_Z1"]))

auc_masked = auc_table["AUC_Z1"].where(auc_table["p_chance_Z1_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))

ggseg.plot_dk(
    data,
    #cmap=mpl.cm.get_cmap("Blues"), 
    cmap=cmap,         # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Cortical AUC Z1",
    vminmax=[0.5,0.8]
)

#data = dict(zip(auc_table["feature_new"], auc_table["AUC_Z2"]))
auc_masked = auc_table["AUC_Z2"].where(auc_table["p_chance_Z2_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))

ggseg.plot_dk(
    data,
    #cmap=mpl.cm.get_cmap("Blues"),
    cmap=cmap,          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Cortical AUC Z2",
    vminmax=[0.5,0.8]
)

#data = dict(zip(auc_table["feature_new"], auc_table["AUC_Z2"]))
auc_masked = auc_table["AUC_slope"].where(auc_table["p_chance_slope_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))

ggseg.plot_dk(
    data,
    #cmap=mpl.cm.get_cmap("Blues"),  
    cmap=cmap,        # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Cortical AUC slope",
    vminmax=[0.5,0.8]
)

cmap = mpl.cm.get_cmap("coolwarm").copy()   # copy so you don't mutate the global cmap
cmap.set_bad("gray")

auc_masked = auc_table["difference"].where(auc_table["p_delong_gain_vs_slope"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))
ggseg.plot_dk(
    data,
    #cmap=mpl.cm.get_cmap("coolwarm"),          # or "coolwarm", "viridis", etc.
    cmap=cmap,
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC gain",
    title="Regions where gain model is better than slope model",
    vminmax=[-0.15,0.15]
)

#%%


