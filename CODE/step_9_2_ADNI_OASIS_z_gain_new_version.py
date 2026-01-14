#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:16:45 2025

@author: johbay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import utilities_life
import utilities_thrive
from utils import SC_idp_cols
from itertools import combinations
import pcntoolkit
from utilities_stats import compare_distributions_by_column
import sys
#%% This file does calculations and makes plot for figure 3
# This file works with the new version of the toolkit

atlas ="DES"
#%%
def load_adni_data(atlas):
    """Load all required ADNI datasets for a given measure."""
    base_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity"
        
    ## Load z-scores and test data
    z_scores_original = pd.read_csv(f"{base_path}/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_{atlas}_all_regions/results/Z_test.csv")
    z_scores = z_scores_original.sort_values(by="observations").reset_index(drop=True)
   
    #test_raw = pd.read_pickle(f'{base_path}/DATA_new_version/{atlas}/test_{atlas}_demented_adults_ADNI.pkl')
    test_raw = pd.read_pickle(f'{base_path}/DATA/{atlas}/test_{atlas}_demented_adults_ADNI.pkl')
    z_scores_full = pd.concat(
    [z_scores.reset_index(drop=True),
        test_raw[["age","ID_subject", "ID_visit", "sex", "site_id", "which_dataset", "site_id2"]].reset_index(drop=True)
    ],
    axis=1
    )
    ## Load clinical data. Wee need this for information such as diagnosis
    ADNI_raw = pd.read_pickle(f"{base_path}/DATA_pre_processed/ADNI_DK_clinical_goodsites.pkl")
    ADNI_adaptation_test = pd.read_pickle(f'{base_path}/DATA_pre_processed/ADNI_DK_adaptation_test.pkl')
 
    ADNI_raw_small = ADNI_raw[["ID_subject", "ID_visit", "DIAGNOSIS"]]
    ADNI_adaptation_test_small = ADNI_adaptation_test[["ID_subject", "ID_visit", "DIAGNOSIS"]]
    
    z_scores_full = z_scores_full.merge(
        ADNI_raw_small,
        on=["ID_subject", "ID_visit"],
        how="left"
        )
    
    z_scores_full = z_scores_full.merge(
        ADNI_adaptation_test_small,
        on=["ID_subject", "ID_visit"],
        how="left",
        suffixes=("", "_new")
        )
    
    z_scores_full["DIAGNOSIS"] = z_scores_full["DIAGNOSIS_new"].combine_first(z_scores_full["DIAGNOSIS"])
    z_scores_full = z_scores_full.drop(columns=["DIAGNOSIS_new"])
    
    z_scores_ADNI = z_scores_full[z_scores_full["DIAGNOSIS"].notna()]
    # OASIS2_raw = pd.read_pickle(f"{base_path}/DATA_pre_processed/OASIS2_long_SC_demented.pkl")[["diagnosis", "ID_visit", "ID", "sex","age"]].rename(columns={"diagnosis": "DIAGNOSIS", "ID": "ID_subject"})
    # OASIS2_raw["Group"] = OASIS2_raw["DIAGNOSIS"]
    
    # OASIS3_raw = pd.read_pickle(f"{base_path}/DATA_pre_processed/OASIS3_long_SC_demented.pkl")[["diagnosis", "ID_visit", "ID_subject", "sex","age"]].rename(columns={"diagnosis": "DIAGNOSIS"})
    # OASIS3_raw["Group"] = OASIS3_raw["DIAGNOSIS"]

    # OASIS2_HC = pd.read_pickle(f"{base_path}/DATA_pre_processed/OASIS2_long_SC.pkl")[["diagnosis", "ID_visit", "ID_subject", "sex","age"]].rename(columns={"diagnosis": "DIAGNOSIS"})

    # OASIS3_HC = pd.read_pickle(f"{base_path}/DATA_pre_processed/OASIS3_long_SC.pkl")[["diagnosis", "ID_visit", "ID_subject", "sex","age"]].rename(columns={"diagnosis": "DIAGNOSIS"})

    
    # #clinical_raw = pd.concat([ADNI_raw, OASIS2_raw, OASIS3_raw], axis=1)
    
    # clinical_raw = pd.concat([ADNI_raw, OASIS2_raw, OASIS3_raw],axis=0)

    
    return z_scores_ADNI

def load_OASIS_data(atlas):
    
    #OASIS2_raw = pd.read_pickle(f"{base_path}/DATA_pre_processed/OASIS2_long_{atlas}_demented.pkl")[["diagnosis", "ID_visit", "ID", "sex","age"]].rename(columns={"diagnosis": "DIAGNOSIS", "ID": "ID_subject"})
    #OASIS2_raw["Group"] = OASIS2_raw["DIAGNOSIS"]

    base_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity"
    
    z_scores_original = pd.read_csv(f"{base_path}/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_{atlas}_all_regions/results/Z_test.csv")
    z_scores = z_scores_original.sort_values(by="observations").reset_index(drop=True)
   
    OASIS3_raw = pd.read_pickle(f"{base_path}/DATA_new_version/{atlas}/OASIS3_long_{atlas}_demented.pkl").rename(columns={"diagnosis": "DIAGNOSIS"})
    OASIS3_raw["Group"] = OASIS3_raw["DIAGNOSIS"]
    
    # OASIS2_HC = pd.read_pickle(f"{base_path}/DATA_pre_processed/OASIS2_long_SC.pkl")[["diagnosis", "ID_visit", "ID_subject", "sex","age"]].rename(columns={"diagnosis": "DIAGNOSIS"})

    OASIS3_HC = pd.read_pickle(f"{base_path}/DATA_new_version/{atlas}/OASIS3_long_{atlas}.pkl").rename(columns={"diagnosis": "DIAGNOSIS"})
    OASIS3_HC["Group"] = OASIS3_HC["DIAGNOSIS"]

  
    test_raw = pd.read_pickle(f'{base_path}/DATA_new_version/{atlas}/test_{atlas}_demented_adults_ADNI.pkl')
    test_raw.reset_index(drop=True, inplace=True)
    
    test_raw_covariates = test_raw[["ID_subject", "site_id", "age", "sex", "ID_visit", "which_dataset", "set", "site_id2"]]

    
    test_raw_covariates['ID_subject'] = test_raw_covariates['ID_subject'].astype('string')
    
    z_scores_full = pd.concat(
        [z_scores, test_raw_covariates],
        axis=1
    )
    
    OASIS3 = pd.concat([OASIS3_HC, OASIS3_raw], axis =0)
    
    OASIS3['ID_subject'] = OASIS3['ID_subject'].astype('string')
    OASIS3['ID_visit'] = OASIS3['ID_visit'].astype('int')
    test_raw_covariates['ID_visit'] = test_raw_covariates['ID_visit'].astype('float').astype('int')

    # make sure the join keys are present on the left
    left_cols = ['ID_subject', 'ID_visit', 'DIAGNOSIS', 'Group', 'age']
    right_cols = ['ID_subject', 'ID_visit', 'site_id2']

    z_scores_OASIS3 = pd.merge(
        OASIS3[left_cols],
        z_scores_full,
        on=['ID_subject', 'ID_visit', 'age'],
        how='inner'
    )
    
    
    z_scores_OASIS3= z_scores_OASIS3[z_scores_OASIS3["site_id2"].notna()].reset_index(drop=True)
    z_scores_OASIS3 = z_scores_OASIS3.rename(columns={"site_id2": "site2"})
   
    return z_scores_OASIS3



def calculate_z_gain(Z2, Z1, age2, age1, R):
    """Compute z-gain between two timepoints."""
    if age2 <= age1:
        raise ValueError(f"age2 must be greater than age1. Got age1={age1}, age2={age2}")
    
    age1 = int(age1)
    age2 = int(age2)
    R = R.toarray()
    r = R[age2, age1]# take from upper triangle
    if r < 0.3:
        raise ValueError(f"Really low r value here: r = {r}")
    sigma = np.sqrt(1 - r**2)
    z_gain = (Z2 - r * Z1) / sigma
    return z_gain

def compute_all_z_gains(df, R, measure):
    """Compute z-gains for all subject pairs of timepoints."""
    results = []
    
    for subject, group in df.groupby("ID_subject"):
        # Sort and remove duplicate ages
        group_sorted = group.sort_values(["age", "ID_visit"])
        group_unique = group_sorted.drop_duplicates(subset="age", keep="first").reset_index(drop=True)
        
        # Calculate gains for all timepoint pairs
        for i, j in combinations(range(len(group_unique)), 2):
            try:
                t1, t2 = i, j
                age1, age2 = group_unique.loc[t1, "age"], group_unique.loc[t2, "age"]
                Z1, Z2 = group_unique.loc[t1, measure], group_unique.loc[t2, measure]
                diag1, diag2 = group_unique.loc[t1, "DIAGNOSIS"], group_unique.loc[t2, "DIAGNOSIS"]
                
                gain = calculate_z_gain(Z2, Z1, age2, age1, R)
                
                results.append({
                    "ID_subject": subject,
                    "age1": age1, "age2": age2,
                    "t1_index": t1, "t2_index": t2,
                    "z_gain": gain,
                    "diagnosis_t1": diag1, "diagnosis_t2": diag2,
                    "Z1": Z1, "Z2": Z2
                })
                
            except Exception as e:
                print(f"Skipped subject {subject} from t1={age1} to t2={age2}: {e}")
                continue
    
    
    return pd.DataFrame(results)

def create_diagnosis_plots(gains, gains_HC, measure, dataset, save=True, save_path="/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/"):
    """Create all diagnosis-related plots."""
    
    # Prepare data
    gains_clean = gains.dropna(subset=["diagnosis_t1", "diagnosis_t2"]).copy()
    gains_clean["diagnosis_t1"] = gains_clean["diagnosis_t1"].replace({2.0: "(E)MCI", 3.0: "AD", 1.0: "HC"})
    gains_clean["diagnosis_t2"] = gains_clean["diagnosis_t2"].replace({2.0: "(E)MCI", 3.0: "AD", 1.0: "HC"})
    gains_clean["diagnosis_transition"] = gains_clean["diagnosis_t1"] + " → " + gains_clean["diagnosis_t2"]
    
    # Color palettes
    custom_palette = {"HC": "#888888", "(E)MCI": "#be311e", "AD": "#4a0004"}
    custom_palette2 = {"AD → AD": "#4a0004", "(E)MCI → AD": "#be311e", "(E)MCI → (E)MCI": "#888888"}
    
    plt.style.use('default')
    sns.set_context('talk')
    
    # Plot 1: Comparison histogram (ADNI vs HC)
    plt.figure(figsize=(8, 5))
    colors = ['#ff424b', '#999999']
    plt.hist([gains["z_gain"], gains_HC["z_gain"]], bins=15, edgecolor='black',
             label=[f"{dataset} Clinical Subjects", "HC Subjects"], alpha=0.7, color=colors)
    plt.title(f"Histogram of Z-Gain: {measure}, {dataset}")
    plt.xlabel("Z-Gain")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    if save: 
        plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/1_{measure}_{dataset}_comparison_hist_new.png")
    plt.show()
    
    # Plot 2: KDE by diagnosis at t2
    plt.figure(figsize=(8, 5))
    for diag in gains_clean["diagnosis_t2"].unique():
        subset = gains_clean[gains_clean["diagnosis_t2"] == diag]
        sns.kdeplot(data=subset, 
                    x="z_gain", 
                    label=str(diag), 
                    fill=True, 
                    alpha=0.6,
                   bw_adjust=1, 
                   linewidth=2, color=custom_palette.get(diag, "gray"))
    
    plt.title(f"Density of Z-Gain by Diagnosis: {measure}, {dataset}")
    plt.xlabel("Z-Gain")
    plt.ylabel("Density")
    plt.grid(False)
    plt.legend(title="Diagnosis")
    plt.tight_layout()
    if save: 
        plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/2_{measure}_{dataset}_kde_diagnosis_new.png")
    plt.show()
    
    
    # Plot 3 Histogram by diagnosis at t2
    plt.figure(figsize=(8, 5))

    for diag in gains_clean["diagnosis_t2"].dropna().unique():
        subset = gains_clean[gains_clean["diagnosis_t2"] == diag]
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
                alpha=0.5
            )
    
    plt.title(f"Histogram of Z-Gain by Diagnosis: {measure}, {dataset}")
    plt.xlabel("Z-Gain")
    plt.ylabel("Density")
    plt.grid(False)
    plt.legend(title="Diagnosis")
    plt.tight_layout()
    
    if save: 
        plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/3_{measure}_{dataset}_hist_diagnosis.png")
    
    plt.show()
    
    # Plot 4: Diagnosis transitions (subset), Density
    subset = gains_clean[gains_clean["diagnosis_transition"].isin(["(E)MCI → AD", "AD → AD", "(E)MCI → (E)MCI"])]
    
    plt.figure(figsize=(8, 5))
    for label, color in custom_palette2.items():
        subset_diag = subset[subset["diagnosis_transition"] == label]
        if not subset_diag.empty:
            sns.kdeplot(data=subset_diag, 
                        x="z_gain", 
                        label=label, 
                        bw_adjust=1,
                        fill=True, 
                        alpha=0.4, 
                        linewidth=2, 
                        color=color)
    
    plt.title(f"Z-Gain by Diagnosis Stability: {measure}, {dataset}")
    plt.xlabel("Z-Gain")
    plt.ylabel("Density")
    plt.legend(title="Diagnosis Transition")
    plt.grid(False)
    plt.tight_layout()
    if save:
        plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/4_{measure}_{dataset}_kde_transitions_new.png")
    plt.show()
    
    
    #Plot 5: Diagnosis transition (subset), Histogram
    
    plt.figure(figsize=(8, 5))

    for label, color in custom_palette2.items():
        subset_diag = subset[subset["diagnosis_transition"] == label]
        if not subset_diag.empty:
            sns.histplot(
                data=subset_diag,
                x="z_gain",
                label=label,
                color=color,
                bins=20,
                common_norm= True,
                stat="density",  # normalize like KDE
                element="step",  # outline style
                fill=True,
                alpha=0.5
            )
    
    plt.title(f"Z-Gain by Diagnosis Stability: {measure}, {dataset}")
    plt.xlabel("Z-Gain")
    plt.ylabel("Density")
    plt.legend(title="Diagnosis Transition")
    plt.grid(False)
    plt.tight_layout()
    
    if save: 
        plt.savefig(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/5_{measure}_{dataset}_hist_transitions_new.png")
    
    plt.show()

 
    return gains_clean

def analyze_single_measure(atlas, idx, measure, save=False, save_path=None):
    """Complete analysis pipeline for a single measure."""
    
    print(f"Analyzing measure: {measure}")
    
    # Load data
    z_scores_ADNI = load_adni_data(atlas)
   
    # Prepare clinical data
    #clinical_ADNI = prepare_clinical_data(z_scores, test_raw, clinical_raw)
    
    z_scores_OASIS3 = load_OASIS_data(atlas)
    

    clinical_ADNI = z_scores_ADNI[z_scores_ADNI["DIAGNOSIS"]!=1]
    HC_ADNI = z_scores_ADNI[z_scores_ADNI["DIAGNOSIS"]==1]
    
    # Load velocity correlation matrix
    velocity_path = f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_{atlas}_all_regions/results/Velocity/"
   
    velocity_data = pd.read_pickle(os.path.join(velocity_path,measure,"velocity_objects.pkl"))
    R = velocity_data['A_sparse_predict']
    
    # Compute z-gains
    gains_clinical_ADNI = compute_all_z_gains(clinical_ADNI, R, measure)
    gains_HC_ADNI = compute_all_z_gains(HC_ADNI, R, measure)
    
    cross_sectional = pd.concat([gains_clinical_ADNI, gains_HC_ADNI])

    # choose where to save it
    save_path = f"{base_dir}/batch_{idx}/cross_sectional.pkl"

    # write the pickle
    with open(save_path, "wb") as f:
         pickle.dump(cross_sectional, f)
    
    
    gains_OASIS3 = compute_all_z_gains(OASIS3, R, measure)
    gains_OASIS3["diagnosis_t1"] = gains_OASIS3["diagnosis_t1"]+1 #make diagnosis coding similar to ADNI
    gains_OASIS3["diagnosis_t2"] = gains_OASIS3["diagnosis_t2"]+1
    
    gains_HC_OASIS3 = gains_OASIS3[(gains_OASIS3["diagnosis_t1"] == 1) & (gains_OASIS3["diagnosis_t2"] == 1)]
    gains_clinical_OASIS3 = gains_OASIS3[~((gains_OASIS3["diagnosis_t1"] == 1) & (gains_OASIS3["diagnosis_t2"] == 1))]
    
    # Create visualizations
    gains_clinical_ADNI = create_diagnosis_plots(gains_clinical_ADNI, gains_HC_ADNI,  measure, "ADNI", save=True)
    
    #test = 
    gains_clinical_OASIS3= create_diagnosis_plots(gains_clinical_OASIS3, gains_HC_OASIS3, measure, "OASIS", save=True)
    
    
    
    # Rename gains_HC back for common standard
    gains_HC_ADNI["diagnosis_t1"] = gains_HC_ADNI["diagnosis_t1"].map({1: "HC"})
    gains_HC_ADNI["diagnosis_t2"] = gains_HC_ADNI["diagnosis_t2"].map({1: "HC"})
    
    gains_HC_ADNI["diagnosis_transition"] = gains_HC_ADNI["diagnosis_t1"] + " → " + gains_HC_ADNI["diagnosis_t2"]
    return gains_clinical_ADNI, gains_HC_ADNI, gains_clinical_OASIS3, gains_HC_OASIS3
# Your actual measure names


def run_prediction_pipeline(
    df: pd.DataFrame,
    measure: str,
    idx: int,
    base_dir: "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_demented_adults_ADNI/",
    prefix: str = "ADNI",
    write_files: bool = True,
    write_preds: bool = True
):
    
    model_path = f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_demented_adults_ADNI/batch_{idx}/Models/NM_0_0_estimate.pkl"
    with open(model_path, 'rb') as f:
        nm = pickle.load(f)
        
    # Prepare input arrays
    X_train = df[["age", "sex"]].to_numpy(dtype=float)
    X_train[:, 0] = X_train[:, 0] / 100  # Normalize age

    Y_train = df[measure].to_numpy(dtype=float)
    Z_train = df["site2"].to_numpy(dtype=float)

    # Create output directory
    dir2 = os.path.join(base_dir, f"batch_{idx}")
    #os.makedirs(dir2, exist_ok=True)
    os.chdir(dir2)

    # File paths
    x_path = f'X_train_SC_test_{prefix}.pkl'
    y_path = f'Y_train_SC_test_{prefix}.pkl'
    z_path = f'trbefile_SC_test_{prefix}.pkl'

    # Save input files
    if write_files:
        with open(x_path, 'wb') as f:
            pickle.dump(pd.DataFrame(X_train), f)
        with open(y_path, 'wb') as f:
            pickle.dump(pd.DataFrame(Y_train), f)
        with open(z_path, 'wb') as f:
            pickle.dump(pd.DataFrame(Z_train), f)

    # Prediction
    z_score, s2 = nm.predict(
        Xs=X_train,
        Y=Y_train,
        tsbefile=z_path,
        alg='hbr',
        likelihood='SHASHb',
        inscaler='standardize',
        outscaler='standardize'
    )

    # Save prediction outputs
    if write_preds:
        with open(f'z_{prefix}_test.pkl', 'wb') as f:
            pickle.dump(pd.DataFrame(z_score), f)
        with open(f's2_{prefix}_test.pkl', 'wb') as f:
            pickle.dump(pd.DataFrame(s2), f)

    return z_score, s2





# Main execution
def main():
    atlas = 'SC'
    save = False  # Set to True if you want to save plots
    save_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/z_pngs/"
    
    measures = SC_idp_cols()
    
    for idx0, measure in enumerate(measures):
        idx = idx0 + 1
        try:
            gains_clinical_ADNI, gains_HC_ADNI, gains_clinical_OASIS3, gains_HC_OASIS3 = analyze_single_measure(atlas, idx, measure, save, save_path)
            print(f"Successfully processed {measure}")
            
            all_gains = pd.concat([gains_clinical_ADNI, gains_HC_ADNI], axis=0)
            with open(f'all_gains_{measure}_new.pkl', 'wb') as f:
                pickle.dump(pd.DataFrame(all_gains), f)
            
            results = compare_distributions_by_column(all_gains, 'z_gain', 'diagnosis_transition', measure_name=measure, alpha=0.05)
            
        except Exception as e:
            print(f"Error processing {measure}: {e}")
            continue

# Run the analysis
if __name__ == "__main__":
    #sys.stdout = open('analysis.txt', 'w')
    main()
    #sys.stdout.close()
    