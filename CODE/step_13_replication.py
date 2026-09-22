#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:39:38 2026

@author: johbay
"""
atlas = "SC"
import pandas as pd
import numpy as np
from pathlib import Path

excluded_UKB = pd.read_csv("/project_cephfs/3022017.06/UKB/qc/T1_ants/sub_exclude_ants_reg.txt", sep="\t", header=None)
excluded_ABCD =pd.read_csv("/project_cephfs/3022017.06/ABCD/qc/T1_ants/ants_pipeline_abcd_rel6.txt", sep="\t", header=None)
excluded_OASIS2 =pd.read_csv("/project_cephfs/3022017.06/OASIS2/qc/T1_ants/sub_exclude_ants_reg.txt", sep="\t", header=None)
excluded_OASIS3 =pd.read_csv("/project_cephfs/3022017.06/OASIS3/qc/T1_ants/sub_exclude_ants_reg.txt", sep="\t", header=None)
excluded_ADNI = pd.read_csv("/project_cephfs/3022017.06/ADNI/qc/T1_ants/sub_exclude_ants_reg.txt", sep="\t", header=None)


test_SC = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC/test_SC_demented_adults_ADNI.pkl")
train_SC = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC/train_SC_demented_adults_ADNI.pkl")
retrain_SC = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC/train_retrain_full_models_SC_adults_ADNI.pkl")

test_DK = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DK/test_DK_demented_adults_ADNI.pkl")
train_DK = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DK/train_DK_demented_adults_ADNI.pkl")
retrain_DK = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DK/train_retrain_full_models_DK_adults_ADNI.pkl")

test_DES = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DES/test_DES_demented_adults_ADNI.pkl")
train_DES = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DES/train_DES_demented_adults_ADNI.pkl")
retrain_DES = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DES/train_retrain_full_models_DES_adults_ADNI.pkl")


#%%

excluded_ABCD=excluded_ABCD[~excluded_ABCD[0].str.contains("ses-6YearFollowUpYArm1", na=False)]
excluded_ABCD["ID_subject"] = excluded_ABCD[0].str.extract(r"(sub-[^/]+)")
excluded_ABCD["ID_visit"] = excluded_ABCD[0].str.extract(r"ses-([^/]+)")

excluded_ABCD=excluded_ABCD.replace({'ID_visit':{'4YearFollowUpYArm1*':'4', 'baselineYear1Arm1*':'1', 
                                   '2YearFollowUpYArm1*':'2', '3YearFollowUpYArm1*':'3'}}, regex = True)

excluded_ABCD["ID_visit"] = excluded_ABCD["ID_visit"].astype(int)

has_scan2 = excluded_UKB[0].str.contains("_scan2", na=False)
excluded_UKB["ID_visit"] = np.where(has_scan2, 2, 1)
excluded_UKB["ID_subject"] = excluded_UKB[0].str.replace("_scan2", "", regex=False)

excluded_OASIS2["ID_subject"] = excluded_OASIS2[0].str.extract(r"(OAS2_\d+)")
excluded_OASIS2["ID_visit"] = excluded_OASIS2[0].str.extract(r"_MR(\d+)").astype(int)

# extract subject ID and the day number
excluded_OASIS3["ID_subject"] = excluded_OASIS3[0].str.extract(r"(OAS3\d+)")
excluded_OASIS3["day"] = excluded_OASIS3[0].str.extract(r"_d(\d+)").astype(int)

# within each subject, rank days ascending → visit 1, 2, 3...
excluded_OASIS3["ID_visit"] = (
    excluded_OASIS3.groupby("ID_subject")["day"]
    .rank(method="dense")
    .astype(int)
)

excluded_ADNI["ID_subject"] = excluded_ADNI[0].str.extract(r"^([^/]+)")
# we just generate a visit ID here. It does not matter - we eclude on Subject ID only
excluded_ADNI["ID_visit"] = 1


parts = [excluded_ABCD, excluded_UKB, excluded_OASIS2, excluded_OASIS3, excluded_ADNI]

excluded = pd.concat(
    [d[["ID_subject", "ID_visit"]] for d in parts],
    ignore_index=True,
)

excluded = excluded[excluded["ID_visit"] != "03A"]
excluded["ID_visit"] = excluded["ID_visit"].astype(int)
#%%

test_SC["ID_visit"] = pd.to_numeric(test_SC["ID_visit"], errors="coerce").astype("Int64")
train_SC["ID_visit"] = pd.to_numeric(train_SC["ID_visit"], errors="coerce").astype("Int64")
retrain_SC["ID_visit"] = pd.to_numeric(retrain_SC["ID_visit"], errors="coerce").astype("Int64")

test_SC.to_pickle(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/{atlas}/test_SC_demented_adults_ADNI.pkl")
train_SC.to_pickle(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/{atlas}/train_SC_demented_adults_ADNI.pkl")
retrain_SC.to_pickle(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/{atlas}/train_retrain_full_models_SC_adults_ADNI.pkl")

test_DK["ID_visit"] = pd.to_numeric(test_DK["ID_visit"], errors="coerce").astype("Int64")
train_DK["ID_visit"] = pd.to_numeric(train_DK["ID_visit"], errors="coerce").astype("Int64")
retrain_DK["ID_visit"] = pd.to_numeric(retrain_DK["ID_visit"], errors="coerce").astype("Int64")

test_DK.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DK/test_DK_demented_adults_ADNI.pkl")
train_DK.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DK/train_DK_demented_adults_ADNI.pkl")
retrain_DK.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DK/train_retrain_full_models_DK_adults_ADNI.pkl")

test_DES["ID_visit"] = pd.to_numeric(test_DES["ID_visit"], errors="coerce").astype("Int64")
train_DES["ID_visit"] = pd.to_numeric(train_DES["ID_visit"], errors="coerce").astype("Int64")
retrain_DES["ID_visit"] = pd.to_numeric(retrain_DES["ID_visit"], errors="coerce").astype("Int64")

test_DES.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DES/test_DES_demented_adults_ADNI.pkl")
train_DES.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DES/train_DES_demented_adults_ADNI.pkl")
retrain_DES.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DES/train_retrain_full_models_DES_adults_ADNI.pkl")

#%%

excluded_pairs = set(zip(excluded["ID_subject"]))

def drop_excluded(df):
    keep = [pair not in excluded_pairs
            for pair in zip(df["ID_subject"])]
    return df[keep]

test_SC_QCed    = drop_excluded(test_SC)
train_SC_QCed   = drop_excluded(train_SC)
retrain_SC_QCed = drop_excluded(retrain_SC)

test_DK_QCed    = drop_excluded(test_DK)
train_DK_QCed   = drop_excluded(train_DK)
retrain_DK_QCed = drop_excluded(retrain_DK)

test_DES_QCed    = drop_excluded(test_DES)
train_DES_QCed   = drop_excluded(train_DES)
retrain_DES_QCed = drop_excluded(retrain_DES)


#write
outdir = Path("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/QCed/DATA/")

train_sites = set(train_SC_QCed["site_id2"].unique())
test_sites  = set(test_SC_QCed["site_id2"].unique())

print("in train but not test:", sorted(train_sites - test_sites))
print("in test but not train:", sorted(test_sites - train_sites))
print("all shared?", train_sites == test_sites)

train_sites = set(train_DK_QCed["site_id2"].unique())
test_sites  = set(test_DK_QCed["site_id2"].unique())

print("in train but not test:", sorted(train_sites - test_sites))
print("in test but not train:", sorted(test_sites - train_sites))
print("all shared?", train_sites == test_sites)

train_sites = set(train_DES_QCed["site_id2"].unique())
test_sites  = set(test_DES_QCed["site_id2"].unique())

print("in train but not test:", sorted(train_sites - test_sites))
print("in test but not train:", sorted(test_sites - train_sites))
print("all shared?", train_sites == test_sites)



dataframes = {
    "test_SC_QCed": test_SC_QCed,
    "train_SC_QCed": train_SC_QCed,
    "retrain_SC_QCed": retrain_SC_QCed,
    "test_DK_QCed": test_DK_QCed,
    "train_DK_QCed": train_DK_QCed,
    "retrain_DK_QCed": retrain_DK_QCed,
    "test_DES_QCed": test_DES_QCed,
    "train_DES_QCed": train_DES_QCed,
    "retrain_DES_QCed": retrain_DES_QCed,
    # add the rest here
}

for name, df in dataframes.items():
    df.to_pickle(outdir / f"{name}.pkl")
    print(f"wrote {name}.pkl  ({df.shape[0]} rows)")
