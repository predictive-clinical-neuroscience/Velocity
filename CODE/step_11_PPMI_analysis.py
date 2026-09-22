

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from step_9_2_ADNI_OASIS_z_gain_new_version import calculate_z_gain, compute_all_z_gains
from utilities_thrive import load_velocity_matrix
import seaborn as sns
from utils import DES_idp_cols, SC_idp_cols, DK_idp_cols
import glob
import os
from utilities_stats import auc_vs_chance_pval, delong_pval
from statsmodels.stats.multitest import multipletests
import numpy as np

#%%
atlas = "SC"
save = True

#%%
data = pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/{atlas}/PPMI_clinical_{atlas}.csv")
z_scores = pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/results/Z_test.csv")

covariate_columns = data[['age', 'sex', 'participant_id', 'visit_id', 'COHORT_DEFINITION', 'site_id2']]
covariate_columns = covariate_columns.rename(columns={"visit_id":"ID_visit","participant_id":"ID_subject","COHORT_DEFINITION":"DIAGNOSIS"})


covariate_columns["age"] = covariate_columns["age"].round()
map_dict = {"BL": 1, "V04": 2, 'V06':3, 'V10':4} #the actual year is encded via age

covariate_columns["ID_visit"] = covariate_columns["ID_visit"].map(map_dict)

#%
PPMI = pd.concat([z_scores, covariate_columns], axis=1)

#%%

if atlas == "SC":
    measures = SC_idp_cols()
    # measures.remove('non-WM-hypointensities')
    # measures.remove('SubCortGrayVol')
    # measures.remove('TotalGrayVol')
    # measures.remove('SupraTentorialVol')
    # measures.remove('SupraTentorialVolNotVent')
    # measures.remove('BrainSegVol-to-eTIV')
    measures.remove('EstimatedTotalIntraCranialVol')
elif atlas =="DES":
    measures = DES_idp_cols()
    # measures.remove('rh_S_orbital_lateral')
    # measures.remove('rh_S_orbital_med-olfact')
    # measures.remove('rh_S_orbital-H_Shaped')
    # 'rh_S_parieto_occipital'
    # 'rh_S_pericallosal'
    # 'rh_S_postcentral'
    # 'rh_S_precentral-inf-part'
    # 'rh_S_precentral-sup-part'
    # 'rh_S_suborbital'
    # 'rh_S_subparietal'
    # 'rh_S_temporal_inf'
    # 'rh_S_temporal_sup'
    # 'rh_S_temporal_transverse'
else:
    measures = DK_idp_cols()

#start_measure = "rh_S_orbital_med-olfact"
#start_idx = measures.index(start_measure)

#%% calculate Z-gains for PPMI data

plot=False
for i,  measure in enumerate(measures):
    #if i >=3: break
    #if i < start_idx+13:
        #continue
    if measure not in PPMI.columns:
        print(f"[skip] {measure}: not a column in PPMI")
        continue
    base_dir = f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Velocity_models/{atlas}_demented/batch_*_{measure}/Velocity/velocity_objects_new_Oct25.pkl"

    hits = glob.glob(base_dir)
    velocity_data = pd.read_pickle(hits[0])
    R = velocity_data['A_sparse_predict']

    gains_PPMI = compute_all_z_gains(PPMI, R, measure= measure)


    custom_palette = {"SWEDD": "#888888", "Prodromal": "#be311e", "Parkinson's Disease": "#4a0004"}

    dataset = "PPMI"
    
    if plot: 
        # Plot 3 Histogram by diagnosis at t2
        plt.figure(figsize=(8, 5))
        
        for diag in gains_PPMI["diagnosis_t2"].dropna().unique():
            subset = gains_PPMI[gains_PPMI["diagnosis_t2"] == diag]
            if not subset.empty:
                sns.histplot(
                    data=subset,
                    x="z_gain",
                    label=str(diag),
                    color=custom_palette.get(diag, "gray"),
                    bins=20,
                    stat="density",  # normalize y-axis
                    element="step",  # outlines instead of bars; use 'poly' for filled
                    common_norm=False,
                    fill=True,
                    alpha=0.5,
                )
        
        plt.title(f"Histogram of Z-Gain by Diagnosis: {measure}, {dataset}")
        plt.xlabel("Z-Gain")
        plt.ylabel("Density")
        plt.xlim(-10,10)
        plt.grid(False)
        plt.legend(title="Diagnosis")
        plt.tight_layout()
        
    if save:
        if plot:
            plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/z_gains/{measure}_{dataset}_hist_diagnosis.png")
        gains_PPMI.to_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/z_gains/{measure}_{dataset}_zgains.csv")
    
    if plot:
        plot.show()

#%% calculate ROC

import sklearn.metrics as metrics
atlas ="SC"
features = SC_idp_cols()


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

# get slopes, which are caclulated in step_14_roc_of_slopes_revision
slopes = pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/slopes_{atlas}_PPMI.csv")


for i, feat in enumerate(features):
    
    path = f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/z_gains/{feat}_PPMI_zgains.csv"
    if not os.path.exists(path):
      print(f"[skip] {feat}: no zgains csv")
      continue
  
    gains_PPMI= pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/z_gains/{feat}_PPMI_zgains.csv")
    gains_PPMI = gains_PPMI[gains_PPMI['diagnosis_t2'] != 'SWEDD']

     
    if feat in non_invert_regions:
        inv_pred = False
    else:
        inv_pred = True
    
    z = gains_PPMI 
    
    z = z.merge(
    slopes[["ID_subject", "age1", "age2", "t1_index", "t2_index", f"slope_{feat}"]],
    on=["ID_subject", "age1", "age2", "t1_index", "t2_index"],
    how="inner",
    )
    # calculate the for and tpr for all thresholds of the classification
    probs = z['z_gain']
    preds = probs > 0
    preds = preds.astype(int)
    
    y_test = z['diagnosis_t2'] == 'Parkinson\'s Disease'
    if inv_pred:
        y_test = 1-y_test.astype(int)
    else:
        y_test = y_test.astype(int)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    auc_gain = metrics.auc(fpr, tpr)
    print('AUC gain: %0.2f' % auc_gain)
    p_chance_gain = auc_vs_chance_pval(probs, y_test)   # <-- added
    probs_gain, y_gain = probs, y_test 
    
    plt.title(f'ROC - {feat} - z_gain')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_gain)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    probs = z['Z1']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = z['diagnosis_t2'] == 'Parkinson\'s Disease'
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
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_z1)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    probs = z['Z2']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = z['diagnosis_t2'] == 'Parkinson\'s Disease'
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
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_z2)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    probs = z[f'slope_{feat}']
    preds = probs > 0
    preds = preds.astype(int)
    y_test = z['diagnosis_t2'] == 'Parkinson\'s Disease'
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
        "p_chance_z_gain": p_chance_gain,
        "p_chance_slope": p_chance_slope,
        "p_chance_z1": p_chance_z1,
        "p_chance_z2": p_chance_z2,
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
    
auc_table.to_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/z_gains/auc_table_PPMI_{atlas}_new.csv", sep='\t')

#%%
# saving these was really hard - I made screen shots
import ggseg
import matplotlib as mpl

auc_table=pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/z_gains/auc_table_PPMI_{atlas}.csv", sep='\t')


auc_table["feature_new"] = (
    auc_table["feature"]
    .str.replace("^L_", "", regex=True)
    .str.replace("^R_", "", regex=True)
    + auc_table["feature"].str[0].map({'L': '_left', 'R': '_right'})
)


auc_table["difference"] = auc_table["AUC_z_gain"] - auc_table["AUC_slope"]
#data = dict(zip(auc_table["feature"], auc_table["AUC_z_gain"]))
auc_masked = auc_table["AUC_z_gain"].where(auc_table["p_chance_z_gain_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))

cmap = mpl.cm.get_cmap("Blues").copy()   # copy so you don't mutate the global cmap
cmap.set_bad("gray")


ggseg.plot_dk(
    data,
    cmap=cmap,          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (z-gain)",
    title="Subcortical AUC (z-gain)",
    vminmax=[0.5,0.7],
    fontsize=18
)

#%

auc_masked = auc_table["AUC_Z1"].where(auc_table["p_chance_z1_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))

ggseg.plot_dk(
    data,
    cmap=cmap,          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC Z1",
    title="Cortical AUC Z1",
    vminmax=[0.5,0.7],
    fontsize=18
)


#
auc_masked = auc_table["AUC_Z2"].where(auc_table["p_chance_z2_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))


ggseg.plot_dk(
    data,
    cmap=cmap,          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC Z2",
    title="Cortical AUC Z2",
    vminmax=[0.5,0.8],
    fontsize=18
)

#%
auc_masked = auc_table["AUC_slope"].where(auc_table["p_chance_slope_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature_new"], auc_masked))


ggseg.plot_dk(
    data,
    cmap=cmap,          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC slope",
    title="Cortical AUC Z2",
    vminmax=[0.5,0.7],
    fontsize=18
)

#%
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
    title="z-gain - slopes ",
    vminmax=[-0.15,0.15]
)

#%%
# import ggseg
# import matplotlib as mpl

# auc_table=pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/SC/z_gains/auc_table_SC2.csv", sep='\t')


# data = dict(zip(auc_table["feature"], auc_table["AUC_z_gain"]))

# ggseg.plot_aseg(
#     data,
#     cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
#     background="w",           # white background
#     edgecolor="k",            # black borders
#     bordercolor="gray",
#     ylabel="AUC (z-gain)",
#     title="Subcortical AUC (z-gain)",
#     vminmax=[0.3,0.9]
# )


# data = dict(zip(auc_table["feature"], auc_table["AUC_Z1"]))
# ggseg.plot_aseg(
#     data,
#     cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
#     background="w",           # white background
#     edgecolor="k",            # black borders
#     bordercolor="gray",
#     ylabel="AUC (z-gain)",
#     title="Subcortical AUC Z1",
#     vminmax=[0.3,0.9]
# )

# data = dict(zip(auc_table["feature"], auc_table["AUC_Z2"]))
# ggseg.plot_aseg(
#     data,
#     cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
#     background="w",           # white background
#     edgecolor="k",            # black borders
#     bordercolor="gray",
#     ylabel="AUC (z-gain)",
#     title="Subcortical AUC Z2",
#     vminmax=[0.3,0.9]
# )

#%%
import ggseg
import matplotlib as mpl
import numpy as np

atlas ="SC"
auc_table = pd.read_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/transfer_models/{atlas}/z_gains/auc_table_PPMI_{atlas}_new.csv", sep='\t')

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
    vminmax=[0.5,0.7]
)

#%
auc_masked = auc_table["AUC_Z1"].where(auc_table["p_chance_z1_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature"], auc_masked))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (Z1)",
    title="Subcortical AUC Z1",
    vminmax=[0.5,0.7]
)

auc_masked = auc_table["AUC_Z2"].where(auc_table["p_chance_z2_fdr"] <= 0.05, np.nan)
data = dict(zip(auc_table["feature"], auc_masked))
ggseg.plot_aseg(
    data,
    cmap=mpl.cm.get_cmap("Blues"),          # or "coolwarm", "viridis", etc.
    background="w",           # white background
    edgecolor="k",            # black borders
    bordercolor="gray",
    ylabel="AUC (Z2)",
    title="Subcortical AUC Z2",
    vminmax=[0.5,0.7]
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
    vminmax=[0.5,0.7]
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


