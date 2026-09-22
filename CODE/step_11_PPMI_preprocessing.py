#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:39:04 2026

@author: johbay
"""

#%% Add PPMI data set
# Male = 1, Female =0
# Diagnosis: 0 = nondemented

import pandas as pd
import matplotlib.pyplot as plt


# DES
data = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/fs7.3.2-aparc.a2009s-thickness.tsv",
                  sep ="\t")

demo = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/tabular_bagel.tsv", sep= "\t")

data = data.rename(columns={"session_id": "visit_id"})

PPMI = data.merge(
    demo[["participant_id", "visit_id", "SEX", "AGE", "COHORT_DEFINITION"]],
    on=["participant_id", "visit_id"],
    how="left"   # or "inner", "right", etc. depending on what you want
)

PPMI.columns = PPMI.columns.str.replace("_thickness", "", regex=False)

# remove thickness from column names
PPMI.columns = PPMI.columns.str.replace("_thickness", "", regex=False)
cols = PPMI.loc[:, "lh_G_and_S_frontomargin":"rh_S_temporal_transverse"].drop(columns="lh_MeanThickness")

PPMI["Mean_Thickness"] = cols.mean(axis=1)
PPMI['Median_Thickness'] = cols.median(axis=1)
# give random site id. should be larger than 100 (number of sites)
PPMI["site_id2"] = "160"
PPMI = PPMI.rename(columns={"AGE":"age", "SEX":"sex"})

PPMI.dropna(inplace=True)

#% plot the distribution of subjects by diagnosis

counts = PPMI["COHORT_DEFINITION"].value_counts()
counts.plot.bar()
plt.show()

#% create a healthy control and clinical set
PPMI_HC = PPMI[PPMI["COHORT_DEFINITION"]=="Healthy Control"].copy()
PPMI_clinical = PPMI[PPMI["COHORT_DEFINITION"]!="Healthy Control"].copy()

map_dict = {0: "F", 1:"M"}
PPMI_HC["sex"] = PPMI_HC["sex"].map(map_dict)

PPMI_clinical["sex"] = PPMI_clinical["sex"].map(map_dict)

#%%
#%
PPMI_clinical.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/PPMI_clinical_DES.csv")
PPMI_HC.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/PPMI_HC_DES.csv")

#%% Desikan Kiliani Atlas
# DK
data = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/fs7.3.2-aparc.DKTatlas-thickness.tsv",
                  sep ="\t")

demo = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/tabular_bagel.tsv", sep= "\t")

data = data.rename(columns={"session_id": "visit_id"})

PPMI = data.merge(
    demo[["participant_id", "visit_id", "SEX", "AGE", "COHORT_DEFINITION"]],
    on=["participant_id", "visit_id"],
    how="left"   # or "inner", "right", etc. depending on what you want
)
#%

# remove thickness from column names
PPMI.columns = PPMI.columns.str.replace("_thickness", "", regex=False)
cols = PPMI.loc[:, "lh_caudalanteriorcingulate":"rh_temporalpole"].drop(columns={"lh_MeanThickness", "rh_MeanThickness"})

PPMI["Mean_Thickness"] = cols.mean(axis=1)
PPMI['Median_Thickness'] = cols.median(axis=1)
# give random site id. should be larger than 100 (number of sites)
PPMI["site_id2"] = "160"
PPMI = PPMI.rename(columns={"AGE":"age", "SEX":"sex"})

PPMI.dropna(inplace=True)
PPMI.columns = PPMI.columns.str.replace("lh_", "L_", regex=False)
PPMI.columns = PPMI.columns.str.replace("rh_", "R_", regex=False)
#% plot the distribution of subjects by diagnosis

counts = PPMI["COHORT_DEFINITION"].value_counts()
counts.plot.bar()
plt.show()

#% create a healthy control and clinical set
PPMI_HC_DK = PPMI[PPMI["COHORT_DEFINITION"]=="Healthy Control"].copy()
PPMI_clinical_DK = PPMI[PPMI["COHORT_DEFINITION"]!="Healthy Control"].copy()

map_dict = {0: "F", 1:"M"}
PPMI_HC_DK["sex"] = PPMI_HC_DK["sex"].map(map_dict)

PPMI_clinical_DK["sex"] = PPMI_clinical_DK["sex"].map(map_dict)
#%%
#%
PPMI_clinical_DK.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/PPMI_clinical_DK.csv")
PPMI_HC_DK.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/PPMI_HC_DK.csv")

#%%
# SC

data = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/fs7.3.2-aseg-volume.tsv",
                  sep ="\t")

demo = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/tabular_bagel.tsv", sep= "\t")

data = data.rename(columns={"session_id": "visit_id"})

PPMI = data.merge(
    demo[["participant_id", "visit_id", "SEX", "AGE", "COHORT_DEFINITION"]],
    on=["participant_id", "visit_id"],
    how="left"   # or "inner", "right", etc. depending on what you want
)
#%

PPMI["site_id2"] = "160"
PPMI = PPMI.rename(columns={"AGE":"age", "SEX":"sex"})

PPMI.dropna(inplace=True)

#% plot the distribution of subjects by diagnosis

counts = PPMI["COHORT_DEFINITION"].value_counts()
counts.plot.bar()
plt.show()

#% create a healthy control and clinical set
PPMI_HC_SC = PPMI[PPMI["COHORT_DEFINITION"]=="Healthy Control"].copy()
PPMI_clinical_SC = PPMI[PPMI["COHORT_DEFINITION"]!="Healthy Control"].copy()

map_dict = {0: "F", 1:"M"}
PPMI_HC_SC["sex"] = PPMI_HC_SC["sex"].map(map_dict)

PPMI_clinical_SC["sex"] = PPMI_clinical_SC["sex"].map(map_dict)
#%%
#%
PPMI_clinical_SC.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/PPMI_clinical_SC.csv")
PPMI_HC_SC.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/FS_DATA/fs_stats-0.2.1/PPMI_HC_SC.csv")

#%% Demographics
