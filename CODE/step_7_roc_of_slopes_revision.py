#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:32:20 2026

@author: johbay

# This file calculates the raw slopes between time points and then calculates ROCs from it
"""

import pandas as pd
from itertools import combinations
import numpy as np
from utils import DK_idp_cols, SC_idp_cols, DES_idp_cols
import os
from statsmodels.stats.multitest import multipletests
import sklearn.metrics as metrics
#from step_6_cacluate_roc import auc_vs_chance_pval
import matplotlib.pyplot as plt

def calculate_slopes(df, measures):
    results = []
    for subject, group in df.groupby("ID_subject"):
        # Sort and remove duplicate ages
        group_sorted = group.sort_values(["age", "ID_visit"])
        group_unique = group_sorted.drop_duplicates(subset="age", keep="first").reset_index(drop=True)

        # Calculate slopes for all timepoint pairs
        for i, j in combinations(range(len(group_unique)), 2):
            try:
                t1, t2 = i, j
                age1, age2 = group_unique.loc[t1, "age"], group_unique.loc[t2, "age"]
                diag1, diag2 = group_unique.loc[t1, "DIAGNOSIS"], group_unique.loc[t2, "DIAGNOSIS"]
                dage = age2 - age1

                row = {
                    "ID_subject": subject,
                    "age1": age1, "age2": age2,
                    "t1_index": t1, "t2_index": t2,
                    "delta_age": dage,
                    "diagnosis_t1": diag1, "diagnosis_t2": diag2,
                }

                # one slope (and its endpoint values) per measure
                for measure in measures:
                    Z1, Z2 = group_unique.loc[t1, measure], group_unique.loc[t2, measure]
                    row[f"slope_{measure}"] = (Z2 - Z1) / dage if dage != 0 else np.nan
                    #row[f"{measure}_t1"] = Z1
                    #row[f"{measure}_t2"] = Z2

                results.append(row)
            except Exception as e:
                pass
    return results

#%%% ADNI
atlas = "DK"
base_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/"
#test_raw = pd.read_pickle(f'{base_path}/DATA/{atlas}/test_{atlas}_demented_adults_ADNI.pkl')

ADNI_raw = pd.read_pickle(f"{base_path}/DATA/{atlas}/ADNI_{atlas}_clinical_goodsites.pkl")

if atlas == "DK":
    measures =DK_idp_cols()
elif atlas == "SC":
    measures = SC_idp_cols()
else:
    measures = DES_idp_cols()

result = calculate_slopes(ADNI_raw, measures)
slopes_df = pd.DataFrame(result)
slopes_df.to_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/slopes_{atlas}.csv")

#%% slopes of PPMI

atlas = "SC"
base_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/"
#test_raw = pd.read_pickle(f'{base_path}/DATA/{atlas}/test_{atlas}_demented_adults_ADNI.pkl')

PPMI_raw = pd.read_csv(f"{base_path}/DATA/{atlas}/PPMI_clinical_{atlas}.csv")

PPMI_raw= PPMI_raw.rename(columns={"visit_id":"ID_visit","participant_id":"ID_subject","COHORT_DEFINITION":"DIAGNOSIS"})

PPMI_raw["age"] = PPMI_raw["age"].round()
map_dict = {"BL": 1, "V04": 2, 'V06':3, 'V10':4} #the actual year is encded via age


PPMI_raw["ID_visit"] = PPMI_raw["ID_visit"].map(map_dict)

#%%

if atlas == "DK":
    measures =DK_idp_cols()
elif atlas == "SC":
    measures = SC_idp_cols()
else:
    measures = DES_idp_cols()

def calculate_slopes(df, measures):
    results = []
    for subject, group in df.groupby("ID_subject"):
        # Sort and remove duplicate ages
        group_sorted = group.sort_values(["age", "ID_visit"])
        group_unique = group_sorted.drop_duplicates(subset="age", keep="first").reset_index(drop=True)

        # Calculate slopes for all timepoint pairs
        for i, j in combinations(range(len(group_unique)), 2):
            try:
                t1, t2 = i, j
                age1, age2 = group_unique.loc[t1, "age"], group_unique.loc[t2, "age"]
                diag1, diag2 = group_unique.loc[t1, "DIAGNOSIS"], group_unique.loc[t2, "DIAGNOSIS"]
                dage = age2 - age1

                row = {
                    "ID_subject": subject,
                    "age1": age1, "age2": age2,
                    "t1_index": t1, "t2_index": t2,
                    "delta_age": dage,
                    "diagnosis_t1": diag1, "diagnosis_t2": diag2,
                }

                # one slope (and its endpoint values) per measure
                for measure in measures:
                    if measure not in group_unique.columns:
                        row[f"slope_{measure}"] = np.nan
                        continue
                    Z1, Z2 = group_unique.loc[t1, measure], group_unique.loc[t2, measure]
                    row[f"slope_{measure}"] = (Z2 - Z1) / dage if dage != 0 else np.nan
                    #row[f"{measure}_t1"] = Z1
                    #row[f"{measure}_t2"] = Z2

                results.append(row)
            except Exception as e:
                pass
    return results

results = calculate_slopes(PPMI_raw, measures)
slopes_df = pd.DataFrame(results)
slopes_df.to_csv(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/slopes_{atlas}_PPMI.csv")
