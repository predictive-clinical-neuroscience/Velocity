#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:24:55 2025

@author: johbay
"""

#ADNI
import pandas as pd
import os
from utils import SC_idp_cols, DES_idp_cols, DK_idp_cols
from utilities_test  import *
from sklearn.model_selection import train_test_split
#%%

FEATURE_FUNCS = {
    "SC": SC_idp_cols,
    "DK": DK_idp_cols,
    "DES": DES_idp_cols
}

data_in_dir  = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/'
#base_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/'
#data_out_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/'
#processing_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity'
#%%

ADNI_aseg = pd.read_csv("/project_cephfs/3022017.06/ADNI/freesurfer/aseg_stats.txt", sep = "\t")
ADNI_lh_DES = pd.read_csv("/project_cephfs/3022017.06/ADNI/freesurfer/lh_aparc_a2009s_stats.txt", sep = "\t")
ADNI_rh_DES = pd.read_csv("/project_cephfs/3022017.06/ADNI/freesurfer/rh_aparc_a2009s_stats.txt", sep = "\t")

ADNI_lh_DK = pd.read_csv("/project_cephfs/3022017.06/ADNI/freesurfer/lh_aparc_stats.txt", sep = "\t")
ADNI_rh_DK = pd.read_csv("/project_cephfs/3022017.06/ADNI/freesurfer/rh_aparc_stats.txt", sep = "\t")

ADNI_lh_DK = ADNI_lh_DK.rename(columns={"lh.aparc.thickness": "ID"})
ADNI_rh_DK = ADNI_rh_DK.rename(columns={"rh.aparc.thickness": "ID"})

ADNI_DK = ADNI_lh_DK.merge(ADNI_rh_DK, on="ID")

ADNI_lh_DES = ADNI_lh_DES.rename(columns={"lh.aparc.a2009s.thickness": "ID"})
ADNI_rh_DES = ADNI_rh_DES.rename(columns={"rh.aparc.a2009s.thickness": "ID"})

ADNI_DES = ADNI_lh_DES.merge(ADNI_rh_DES, on="ID")

ADNI_DES = ADNI_DES.loc[:, ~ADNI_DES.columns.str.contains("eTIV")]
ADNI_DES = ADNI_DES.loc[:, ~ADNI_DES.columns.str.contains("BrainSegVolNotVent")]

ADNI_DK = ADNI_DK.loc[:, ~ADNI_DK.columns.str.contains("eTIV")]
ADNI_DK = ADNI_DK.loc[:, ~ADNI_DK.columns.str.contains("BrainSegVolNotVent")]

ADNI_aseg = ADNI_aseg.rename(columns={"Measure:volume": "ID"})
# we need all variables that were processed using longitudian freesurfer
ADNI_aseg_long = ADNI_aseg[ADNI_aseg["ID"].str.contains(r"\.long\.", regex=True)].copy()
ADNI_DES_long = ADNI_DES[ADNI_DES["ID"].str.contains(r"\.long\.", regex=True)].copy()
ADNI_DK_long = ADNI_DK[ADNI_DK["ID"].str.contains(r"\.long\.", regex=True)]


pattern = r'_I[^_]*'

ADNI_aseg_long["IID"] = ADNI_aseg_long["ID"].str.extract(r'_((I[^.]*))').iloc[:, 0]
ADNI_DES_long["IID"] = ADNI_DES_long["ID"].str.extract(r'_((I[^.]*))').iloc[:, 0]
ADNI_DK_long["IID"] = ADNI_DK_long["ID"].str.extract(r'_((I[^.]*))').iloc[:, 0]

ADNI_DES_long = ADNI_DES_long[~ADNI_DES_long['ID'].str.contains("real", na=False)]
ADNI_DK_long = ADNI_DK_long[~ADNI_DK_long['ID'].str.contains("real", na=False)]
ADNI_aseg_long = ADNI_aseg_long[~ADNI_aseg_long["IID"].str.contains("real", na=False)]

ADNI_aseg_long["IID"] = ADNI_aseg_long["IID"].astype(str)
#%%
ADNIMERGE = pd.read_csv('/project_cephfs/3022017.06/ADNI/phenotypes/ADNIMERGE_02May2025.csv')
ADNIMERGE = ADNIMERGE.rename(columns={'PTID': 'Subject', 'VISCODE': 'VISCODE2'})

ADNI_conversion_info = pd.read_csv('/project_cephfs/3022017.06/ADNI/phenotypes/All_Subjects_DXSUM_02May2025.csv')
ADNI_conversion_info = ADNI_conversion_info[["PTID", "VISCODE", "VISCODE2", "DIAGNOSIS"]]
ADNI_conversion_info = ADNI_conversion_info.rename(columns={'PTID': 'Subject'})

ADNI_demo1 = pd.read_csv('/project_cephfs/3022017.06/ADNI/phenotypes/Velocity_4_04_2025.csv')
ADNI_demo2 = pd.read_csv('/project_cephfs/3022017.06/ADNI/phenotypes/Velocity_MP-RAGE_4_04_2025.csv')
ADNI_demo = pd.concat([ADNI_demo1, ADNI_demo2], axis = 0)
ADNI_demo.rename(columns={"Image Data ID": "IID", "Visit":"VISCODE"}, inplace ="True")
ADNI_demo["Sex"] = ADNI_demo["Sex"].replace({"M": 1, "F": 0})


MRI_info_1 = pd.read_csv('/project_cephfs/3022017.06/ADNI/phenotypes/MRI3META_19May2025.csv')
MRI_info_1= MRI_info_1[["PTID", "VISCODE", "VISCODE2"]]
MRI_info_2 = pd.read_csv('/project_cephfs/3022017.06/ADNI/phenotypes/MRIMETA_19May2025.csv')
MRI_info_2= MRI_info_2[["PTID", "VISCODE", "VISCODE2"]]

MRI_info = pd.concat([MRI_info_1, MRI_info_2], axis =0)
MRI_info.rename(columns={"PTID": "Subject"}, inplace=True) 

ADNI_demo = ADNI_demo.merge(MRI_info, on=['Subject', 'VISCODE'], how ="left")
#%%
ADNI_demo_full = ADNI_demo.merge( ADNI_conversion_info[["Subject", "VISCODE2","DIAGNOSIS"]],
                                 on=['Subject', 'VISCODE2'], 
                                 how='left')
#remove werid ID
ADNI_demo_full = ADNI_demo_full[ADNI_demo_full['VISCODE2'] != 'uns1']

#ADNI_demo_full = ADNI_demo_full[~ADNI_demo_full["Visit"].str.contains("sc", na=False)]



# # convert weird mXX IDs into numbers
# ADNI_demo_full['ID_visit'] = (
#     ADNI_demo_full.groupby('Subject')['VISCODE2']
#     .rank(method='dense')
#     .astype(int)
#  )


# ADNI_demo_full["VISCODE2"].unique()


#%%
ADNI_DK_long = ADNI_DK_long.merge(ADNI_demo_full, on ="IID", how="left")
ADNI_DES_long = ADNI_DES_long.merge(ADNI_demo_full, on ="IID", how="left")
ADNI_aseg_long = ADNI_aseg_long.merge(ADNI_demo_full, on ="IID", how ="left")


ADNI_DK_long = ADNI_DK_long.merge(
    ADNIMERGE[["Subject", "VISCODE2", "SITE"]],
    on=["Subject", "VISCODE2"],
    how="left"
)

ADNI_DES_long = ADNI_DES_long.merge(
    ADNIMERGE[["Subject", "VISCODE2", "SITE"]],
    on=["Subject", "VISCODE2"],
    how="left"
)

ADNI_aseg_long = ADNI_aseg_long.merge(
    ADNIMERGE[["Subject", "VISCODE2", "SITE"]],
    on=["Subject", "VISCODE2"],
    how="left"
)

ADNI_aseg_long["VISCODE2"] =ADNI_aseg_long["VISCODE2"].astype(str)
ADNI_aseg_long = ADNI_aseg_long[~ADNI_aseg_long['VISCODE2'].str.contains("scmri")]
ADNI_aseg_long = ADNI_aseg_long[~ADNI_aseg_long['VISCODE2'].str.contains("sc")]

ADNI_DK_long["VISCODE2"] =ADNI_DK_long["VISCODE2"].astype(str)
ADNI_DK_long = ADNI_DK_long[~ADNI_DK_long['VISCODE2'].str.contains("scmri")]
ADNI_DK_long = ADNI_DK_long[~ADNI_DK_long['VISCODE2'].str.contains("sc")]

ADNI_DES_long["VISCODE2"] =ADNI_DES_long["VISCODE2"].astype(str)
ADNI_DES_long = ADNI_DES_long[~ADNI_DES_long['VISCODE2'].str.contains("scmri")]
ADNI_DES_long = ADNI_DES_long[~ADNI_DES_long['VISCODE2'].str.contains("sc")]


#diagnosis_counts = ADNI_DES_long.groupby("Subject")["DIAGNOSIS"].nunique()

# subset adapattaion and test set.
#ADNI_DK_adaptation = ADNI_DK_long[ADNI_DK_long["DIAGNOSIS"] == 1]
#ADNI_DK_clinical = ADNI_DK_long[ADNI_DK_long["DIAGNOSIS"] != 1]


#idp_cols = DES_idp_cols()

#%% demographics
ADNI_aseg_long = ADNI_aseg_long[~ADNI_aseg_long['ID'].str.contains("real")]
ADNI_DES_long = ADNI_DES_long[~ADNI_DES_long['ID'].str.contains("real")]
ADNI_DK_long = ADNI_DK_long[~ADNI_DK_long['ID'].str.contains("real")]


ADNI_SC_adaptation = ADNI_aseg_long[ADNI_aseg_long["Group"] == "CN"]
ADNI_SC_clinical = ADNI_aseg_long[ADNI_aseg_long["Group"] != "CN"]

ADNI_DK_adaptation = ADNI_DK_long[ADNI_DK_long["Group"] == "CN"]
ADNI_DK_clinical = ADNI_DK_long[ADNI_DK_long["Group"] != "CN"]

ADNI_DES_adaptation = ADNI_DES_long[ADNI_DES_long["SITE"] != "CN"]
ADNI_DES_clinical = ADNI_DES_long[ADNI_DES_long["Group"] != "CN"]

#%%

ADNI_SC_adaptation = ADNI_SC_adaptation.dropna(subset=['SITE'])
ADNI_SC_clinical = ADNI_SC_clinical.dropna(subset=['SITE'])

ADNI_DK_adaptation = ADNI_DK_adaptation.dropna(subset=['SITE'])
ADNI_DK_clinical = ADNI_DK_clinical.dropna(subset=['SITE'])

ADNI_DES_adaptation = ADNI_DES_adaptation.dropna(subset=['SITE'])
ADNI_DES_clinical = ADNI_DES_clinical.dropna(subset=['SITE'])

#%%

len(ADNI_SC_adaptation["Subject"].unique())
len(ADNI_SC_clinical["Subject"].unique())

len(ADNI_SC_adaptation["SITE"].unique())
len(ADNI_SC_clinical["SITE"].unique())

ADNI_SC_adaptation.groupby("Sex").count()/len(ADNI_SC_adaptation)
ADNI_SC_clinical.groupby("Sex").count()/len(ADNI_SC_clinical)

ADNI_SC_adaptation["Age"].mean()
ADNI_SC_adaptation["Age"].std()

ADNI_SC_clinical["Age"].mean()
ADNI_SC_clinical["Age"].std()

len(ADNI_SC_adaptation["SITE"].unique())
len(ADNI_SC_clinical["SITE"].unique())

table = pd.crosstab(ADNI_aseg_long["SITE"], ADNI_aseg_long["Group"])
valid_sites = table[table["CN"] != 0].index
ADNI_SC_clinical = ADNI_SC_clinical[ADNI_SC_clinical["SITE"].isin(valid_sites)]

table = pd.crosstab(ADNI_DK_long["SITE"], ADNI_DK_long["Group"])
valid_sites = table[table["CN"] != 0].index
ADNI_DK_clinical = ADNI_DK_clinical[ADNI_DK_clinical["SITE"].isin(valid_sites)]

table = pd.crosstab(ADNI_DES_long["SITE"], ADNI_DES_long["Group"])
valid_sites = table[table["CN"] != 0].index
ADNI_DES_clinical = ADNI_DES_clinical[ADNI_DES_clinical["SITE"].isin(valid_sites)]

#%%
site_map = {site: i + 1 for i, site in enumerate(sorted(ADNI_SC_adaptation["SITE"].unique()))}

ADNI_SC_adaptation['site2'] = ADNI_SC_adaptation['SITE'].map(site_map)
ADNI_SC_clinical['site2'] = ADNI_SC_clinical['SITE'].map(site_map)

ADNI_DK_adaptation['site2'] = ADNI_DK_adaptation['SITE'].map(site_map)
ADNI_DK_clinical['site2'] = ADNI_DK_clinical['SITE'].map(site_map)

ADNI_DES_adaptation['site2'] = ADNI_DES_adaptation['SITE'].map(site_map)
ADNI_DES_clinical['site2'] = ADNI_DES_clinical['SITE'].map(site_map)
#%%
ADNI_SC_adaptation= ADNI_SC_adaptation.rename(columns={"Sex": "sex", "Age": "age", "Subject":"ID_subject", "SITE":"site_id"})
ADNI_SC_clinical = ADNI_SC_clinical.rename(columns={"Sex": "sex", "Age":"age", "Subject":"ID_subject", "SITE":"site_id"})
#%%
ADNI_DK_adaptation= ADNI_DK_adaptation.rename(columns={"Sex": "sex", "Age": "age", "Subject":"ID_subject", "SITE":"site_id"})
ADNI_DK_clinical = ADNI_DK_clinical.rename(columns={"Sex": "sex", "Age":"age", "Subject":"ID_subject", "SITE":"site_id"})

ADNI_DES_adaptation= ADNI_DES_adaptation.rename(columns={"Sex": "sex", "Age": "age", "Subject":"ID_subject", "SITE":"site_id"})
ADNI_DES_clinical = ADNI_DES_clinical.rename(columns={"Sex": "sex", "Age":"age", "Subject":"ID_subject", "SITE":"site_id"})


#%%
atlas = "DK"
features = FEATURE_FUNCS[atlas]()
ADNI_DK_adaptation.columns = ADNI_DK_adaptation.columns.str.replace("lh", "L", regex=True)
ADNI_DK_adaptation.columns = ADNI_DK_adaptation.columns.str.replace("rh", "R", regex=True)
ADNI_DK_adaptation.columns = ADNI_DK_adaptation.columns.str.replace("_thickness", "", regex=True)
ADNI_DK_adaptation.columns = ADNI_DK_adaptation.columns.str.replace("_thickness", "", regex=True)


ADNI_DK_clinical.columns = ADNI_DK_clinical.columns.str.replace("lh", "L", regex=True)
ADNI_DK_clinical.columns = ADNI_DK_clinical.columns.str.replace("rh", "R", regex=True)
ADNI_DK_clinical.columns = ADNI_DK_clinical.columns.str.replace("_thickness", "", regex=True)
ADNI_DK_clinical.columns = ADNI_DK_clinical.columns.str.replace("_thickness", "", regex=True)

ADNI_DK_clinical = ADNI_DK_clinical.rename(columns={"R_entoRinal":"R_entorhinal", "L_entoRinal":"L_entorhinal"})
ADNI_DK_clinical = ADNI_DK_clinical.drop(["R_MeanThickness", "L_MeanThickness"], axis=1)
ADNI_DK_clinical["Median_Thickness"] = ADNI_DK_clinical[features].median(axis=1)
ADNI_DK_clinical["Mean_Thickness"] = ADNI_DK_clinical[features].mean(axis=1)

ADNI_DK_adaptation = ADNI_DK_adaptation.rename(columns={"R_entoRinal":"R_entorhinal", "L_entoRinal":"L_entorhinal"})
ADNI_DK_adaptation = ADNI_DK_adaptation.drop(["R_MeanThickness", "L_MeanThickness"], axis=1)
ADNI_DK_adaptation["Median_Thickness"] = ADNI_DK_adaptation[features].median(axis=1)
ADNI_DK_adaptation["Mean_Thickness"] = ADNI_DK_adaptation[features].mean(axis=1)
#%%
atlas = "DES"
features = FEATURE_FUNCS[atlas]()

ADNI_DES_adaptation.columns = ADNI_DES_adaptation.columns.str.replace("_thickness", "", regex=True)
ADNI_DES_adaptation.columns = ADNI_DES_adaptation.columns.str.replace("_thickness", "", regex=True)


ADNI_DES_clinical.columns = ADNI_DES_clinical.columns.str.replace("_thickness", "", regex=True)
ADNI_DES_clinical.columns = ADNI_DES_clinical.columns.str.replace("_thickness", "", regex=True)

ADNI_DES_clinical = ADNI_DES_clinical.rename(columns={"rh_entoRinal":"rh_entorhinal", "lh_entoRinal":"lh_entorhinal"})
ADNI_DES_clinical = ADNI_DES_clinical.drop(["rh_MeanThickness", "lh_MeanThickness"], axis=1)
ADNI_DES_clinical["Median_Thickness"] = ADNI_DES_clinical[features].median(axis=1)
ADNI_DES_clinical["Mean_Thickness"] = ADNI_DES_clinical[features].mean(axis=1)

ADNI_DES_adaptation = ADNI_DES_adaptation.rename(columns={"rh_entoRinal":"rh_entorhinal", "lh_entoRinal":"lh_entorhinal"})
ADNI_DES_adaptation = ADNI_DES_adaptation.drop(["rh_MeanThickness", "lh_MeanThickness"], axis=1)
ADNI_DES_adaptation["Median_Thickness"] = ADNI_DES_adaptation[features].median(axis=1)
ADNI_DES_adaptation["Mean_Thickness"] = ADNI_DES_adaptation[features].mean(axis=1)


#%%
atlas = "SC"
features = FEATURE_FUNCS[atlas]()


ADNI_SC_adaptation.to_pickle(os.path.join(data_in_dir, "SC","ADNI_SC_adaptation.pkl"))
ADNI_SC_clinical.to_pickle(os.path.join(data_in_dir, "SC","ADNI_SC_clinical.pkl"))
ADNI_DES_adaptation.to_pickle(os.path.join(data_in_dir, "DES", "ADNI_DES_adaptation.pkl"))
ADNI_DES_clinical.to_pickle(os.path.join(data_in_dir, "DES","ADNI_DES_clinical.pkl"))
ADNI_DK_adaptation.to_pickle(os.path.join(data_in_dir, "DK","ADNI_DK_adaptation.pkl"))
ADNI_DK_clinical.to_pickle(os.path.join(data_in_dir, "DK","ADNI_DK_clinical.pkl"))
#%%
ADNI_SC_adaptation.to_csv(os.path.join(data_in_dir, "SC","csv/ADNI_SC_adaptation.csv"), index=False)
ADNI_SC_clinical.to_csv(os.path.join(data_in_dir, "SC","csv/ADNI_SC_clinical.csv"), index=False)
ADNI_DES_adaptation.to_csv(os.path.join(data_in_dir,"DES",  "csv/ADNI_DES_adaptation.csv"), index=False)
ADNI_DES_clinical.to_csv(os.path.join(data_in_dir, "DES", "csv/ADNI_DES_clinical.csv"), index=False)
ADNI_DK_adaptation.to_csv(os.path.join(data_in_dir, "DK", "csv/ADNI_DK_adaptation.csv"), index=False)
ADNI_DK_clinical.to_csv(os.path.join(data_in_dir, "DK", "csv/ADNI_DK_clinical.csv"), index=False)

#%%

ADNI_SC_adaptation = pd.read_pickle(os.path.join(data_in_dir, "SC", "ADNI_SC_adaptation.pkl"))
ADNI_SC_clinical = pd.read_pickle(os.path.join(data_in_dir, "SC","ADNI_SC_clinical.pkl"))

ADNI_DK_adaptation = pd.read_pickle(os.path.join(data_in_dir, "DK", "ADNI_DK_adaptation.pkl"))
ADNI_DK_clinical = pd.read_pickle(os.path.join(data_in_dir, "DK", "ADNI_DK_clinical.pkl"))


ADNI_DES_adaptation = pd.read_pickle(os.path.join(data_in_dir, "DES","ADNI_DES_adaptation.pkl"))
ADNI_DES_clinical = pd.read_pickle(os.path.join(data_in_dir, "DES", "ADNI_DES_clinical.pkl"))

ADNI_DES_clinical['Group'].value_counts()
ADNI_DES_clinical.groupby('Group')['ID_subject'].nunique()
#%% SC
sex_site_table = pd.crosstab(ADNI_SC_adaptation['site2'], ADNI_SC_adaptation['sex'])
valid_sites = sex_site_table[(sex_site_table > 2).all(axis=1)].index

ADNI_SC_adaptation= ADNI_SC_adaptation[ADNI_SC_adaptation['site2'].isin(valid_sites)]
ADNI_SC_adaptation["strata"] = ADNI_SC_adaptation["sex"].astype(str) + "_" + ADNI_SC_adaptation["site2"].astype(str)

ADNI_SC_clinical_goodsites = ADNI_SC_clinical[ADNI_SC_clinical['site2'].isin(valid_sites)]

# Step: generate visit ID based on age rank
ADNI_SC_adaptation = ADNI_SC_adaptation.sort_values(by=['ID_subject', 'age'])
ADNI_SC_adaptation['ID_visit'] = ADNI_SC_adaptation.groupby('ID_subject')['age'].rank(method='first').astype(int)

ADNI_SC_clinical_goodsites = ADNI_SC_clinical_goodsites.sort_values(by=['ID_subject', 'age'])
ADNI_SC_clinical_goodsites['ID_visit'] = ADNI_SC_clinical_goodsites.groupby('ID_subject')['age'].rank(method='first').astype(int)
#%%
ADNI_SC_clinical_goodsites['site_id'] = ADNI_SC_clinical_goodsites['site_id']+900
ADNI_SC_adaptation['site_id'] = ADNI_SC_adaptation['site_id']+900
#%%
subjects_df = (
    ADNI_SC_adaptation
    .groupby('ID_subject')
    .agg({
        'sex': 'first',
        'site_id': 'first'
    })
    .reset_index()
)


df_train, df_test = train_test_split(
    ADNI_SC_adaptation,
    test_size=0.5,             # or any other proportion
    stratify=ADNI_SC_adaptation['strata'],
    random_state=42
)

pd.crosstab(df_train['site2'], df_train['sex'])
pd.crosstab(df_test['site2'], df_test['sex'])


assert_few_nans(ADNI_SC_adaptation)
assert_sex_column_is_binary(ADNI_SC_adaptation)
assert_no_negatives(ADNI_SC_adaptation, features)
assert_few_nans(ADNI_SC_clinical)
assert_sex_column_is_binary(ADNI_SC_clinical)
assert_no_negatives(ADNI_SC_clinical, features)

df_train.to_pickle(os.path.join(data_in_dir, "SC","ADNI_SC_adaptation_train.pkl"))
ADNI_SC_clinical_goodsites.to_pickle(os.path.join(data_in_dir, "SC","ADNI_SC_clinical_goodsites.pkl"))
df_test.to_pickle(os.path.join(data_in_dir, "SC","ADNI_SC_adaptation_test.pkl"))

#%%
#%% DK
sex_site_table = pd.crosstab(ADNI_DK_adaptation['site2'], ADNI_DK_adaptation['sex'])
valid_sites = sex_site_table[(sex_site_table > 2).all(axis=1)].index

ADNI_DK_adaptation= ADNI_DK_adaptation[ADNI_DK_adaptation['site2'].isin(valid_sites)]
ADNI_DK_adaptation["strata"] = ADNI_DK_adaptation["sex"].astype(str) + "_" + ADNI_DK_adaptation["site2"].astype(str)

ADNI_DK_clinical_goodsites = ADNI_DK_clinical[ADNI_DK_clinical['site2'].isin(valid_sites)]

# Step: generate visit ID based on age rank
ADNI_DK_adaptation = ADNI_DK_adaptation.sort_values(by=['ID_subject', 'age'])
ADNI_DK_adaptation['ID_visit'] = ADNI_DK_adaptation.groupby('ID_subject')['age'].rank(method='first').astype(int)

ADNI_DK_clinical_goodsites = ADNI_DK_clinical_goodsites.sort_values(by=['ID_subject', 'age'])
ADNI_DK_clinical_goodsites['ID_visit'] = ADNI_DK_clinical_goodsites.groupby('ID_subject')['age'].rank(method='first').astype(int)
#%%
ADNI_DK_clinical_goodsites['site_id'] = ADNI_DK_clinical_goodsites['site_id']+900
ADNI_DK_adaptation['site_id'] = ADNI_DK_adaptation['site_id']+900
#%%
subjects_df = (
    ADNI_DK_adaptation
    .groupby('ID_subject')
    .agg({
        'sex': 'first',
        'site_id': 'first'
    })
    .reset_index()
)


df_train, df_test = train_test_split(
    ADNI_DK_adaptation,
    test_size=0.5,             # or any other proportion
    stratify=ADNI_DK_adaptation['strata'],
    random_state=42
)

pd.crosstab(df_train['site2'], df_train['sex'])
pd.crosstab(df_test['site2'], df_test['sex'])

features = DK_idp_cols()
assert_few_nans(ADNI_DK_adaptation)
assert_sex_column_is_binary(ADNI_DK_adaptation)
assert_no_negatives(ADNI_DK_adaptation, features)
assert_few_nans(ADNI_DK_clinical)
assert_sex_column_is_binary(ADNI_DK_clinical)
assert_no_negatives(ADNI_DK_clinical, features)

df_train.to_pickle(os.path.join(data_in_dir, "DK","ADNI_DK_adaptation_train.pkl"))
ADNI_DK_clinical_goodsites.to_pickle(os.path.join(data_in_dir, "DK","ADNI_DK_clinical_goodsites.pkl"))
df_test.to_pickle(os.path.join(data_in_dir, "DK","ADNI_DK_adaptation_test.pkl"))
#%%
#%% DES
sex_site_table = pd.crosstab(ADNI_DES_adaptation['site2'], ADNI_DES_adaptation['sex'])
valid_sites = sex_site_table[(sex_site_table > 2).all(axis=1)].index

ADNI_DES_adaptation= ADNI_DES_adaptation[ADNI_DES_adaptation['site2'].isin(valid_sites)]
ADNI_DES_adaptation["strata"] = ADNI_DES_adaptation["sex"].astype(str) + "_" + ADNI_DES_adaptation["site2"].astype(str)

ADNI_DES_clinical_goodsites = ADNI_DES_clinical[ADNI_DES_clinical['site2'].isin(valid_sites)]

# Step: generate visit ID based on age rank
ADNI_DES_adaptation = ADNI_DES_adaptation.sort_values(by=['ID_subject', 'age'])
ADNI_DES_adaptation['ID_visit'] = ADNI_DES_adaptation.groupby('ID_subject')['age'].rank(method='first').astype(int)

ADNI_DES_clinical_goodsites = ADNI_DES_clinical_goodsites.sort_values(by=['ID_subject', 'age'])
ADNI_DES_clinical_goodsites['ID_visit'] = ADNI_DES_clinical_goodsites.groupby('ID_subject')['age'].rank(method='first').astype(int)
#%%
ADNI_DES_clinical_goodsites['site_id'] = ADNI_DES_clinical_goodsites['site_id']+900
ADNI_DES_adaptation['site_id'] = ADNI_DES_adaptation['site_id']+900
#%%
subjects_df = (
    ADNI_DES_adaptation
    .groupby('ID_subject')
    .agg({
        'sex': 'first',
        'site_id': 'first'
    })
    .reset_index()
)


df_train, df_test = train_test_split(
    ADNI_DES_adaptation,
    test_size=0.5,             # or any other proportion
    stratify=ADNI_DES_adaptation['strata'],
    random_state=42
)

pd.crosstab(df_train['site2'], df_train['sex'])
pd.crosstab(df_test['site2'], df_test['sex'])

features = DES_idp_cols()
assert_few_nans(ADNI_DES_adaptation)
assert_sex_column_is_binary(ADNI_DES_adaptation)
assert_no_negatives(ADNI_DES_adaptation, features)
assert_few_nans(ADNI_DES_clinical)
assert_sex_column_is_binary(ADNI_DES_clinical)
assert_no_negatives(ADNI_DES_clinical, features)

df_train.to_pickle(os.path.join(data_in_dir, "DES","ADNI_DES_adaptation_train.pkl"))
ADNI_DES_clinical_goodsites.to_pickle(os.path.join(data_in_dir, "DES", "ADNI_DES_clinical_goodsites.pkl"))
df_test.to_pickle(os.path.join(data_in_dir, "DES", "ADNI_DES_adaptation_test.pkl"))





#%%
# txfer_output_path = os.path.join(processing_dir,'Transfer_SHASHb_1_SC_demented_adults_adapt_balanced')
# os.makedirs(txfer_output_path, exist_ok=True)



# sex_site_table = pd.crosstab(ADNI_SC_adaptation['site2'], ADNI_SC_adaptation['Sex'])
# valid_sites = sex_site_table[(sex_site_table > 10).all(axis=1)].index

# ADNI_SC_adaptation= ADNI_SC_adaptation[ADNI_SC_adaptation['site2'].isin(valid_sites)]
# ADNI_SC_adaptation["strata"] = ADNI_SC_adaptation["Sex"].astype(str) + "_" + ADNI_SC_adaptation["site2"].astype(str)


# df_train, df_test = train_test_split(
#     ADNI_SC_adaptation,
#     test_size=0.5,             # or any other proportion
#     stratify=ADNI_SC_adaptation['strata'],
#     random_state=42
# )

# features = SC_idp_cols()
# #features.extend(["Median_Thickness", "Mean_Thickness"])
# missing_sites = set(df_train['site2']) - set(df_test['site2'])
# print("Sites in df_train but not in df_test:", missing_sites)


# Y_adapt = df_train[features].to_numpy(dtype=float)
# Y_test_txfr = df_test[features].to_numpy(dtype=float)

# X_adapt = df_train[['Age','Sex']]
# X_adapt['Age'] = (X_adapt['Age']/100)
# X_adapt = X_adapt.to_numpy(dtype=float)
# batch_effects_adapt = df_train[['site2']].to_numpy(dtype=int)
# batch_effects_adapt = batch_effects_adapt + 200

# X_test_txfr = df_test[['Age','Sex']]
# X_test_txfr['Age'] = (X_test_txfr['Age']/100)
# X_test_txfr = X_test_txfr.to_numpy(dtype=float)
# batch_effects_test_txfr = df_test[['site2']].to_numpy(dtype=int)
# batch_effects_test_txfr = batch_effects_test_txfr + 200

# # features = DK_idp_cols()
# # features.extend(["Median_Thickness", "Mean_Thickness"])

# # Y_adapt = ADNI_DK_adaptation[features].to_numpy(dtype=float)
# # Y_test_txfr = ADNI_DK_clinical[features].to_numpy(dtype=float)

# # X_adapt = ADNI_DK_adaptation[['Age','Sex']]
# # X_adapt['Age'] = (X_adapt['Age']/100)
# # X_adapt = X_adapt.to_numpy(dtype=float)
# # batch_effects_adapt = ADNI_DK_adaptation[['site2']].to_numpy(dtype=int)


# #%%SC
# X_adapt = ADNI_SC_adaptation[['Age','Sex']]
# X_adapt['Age'] = (X_adapt['Age']/100)
# X_adapt = X_adapt.to_numpy(dtype=float)
# batch_effects_adapt = ADNI_SC_adaptation[['site2']].to_numpy(dtype=int)
# #batch_effects_adapt = batch_effects_adapt + 200

# #%%
# with open(os.path.join(txfer_output_path,'X_adaptation.pkl'), 'wb') as file:
#     pickle.dump(pd.DataFrame(X_adapt), file)
# with open(os.path.join(txfer_output_path,'Y_adaptation.pkl'), 'wb') as file:
#     pickle.dump(pd.DataFrame(Y_adapt), file) 
# with open(os.path.join(txfer_output_path,'adbefile.pkl'), 'wb') as file:
#     pickle.dump(pd.DataFrame(batch_effects_adapt), file) 

# #%% SC

# X_test_txfr = ADNI_SC_adaptation[['Age','Sex']]
# X_test_txfr['Age'] = (X_test_txfr['Age']/100)
# X_test_txfr = X_test_txfr.to_numpy(dtype=float)
# batch_effects_test_txfr = ADNI_SC_adaptation[['site2']].to_numpy(dtype=int)
# batch_effects_test_txfr = batch_effects_test_txfr + 200

# #%% DK
# X_test_txfr = ADNI_DK_clinical[['Age','Sex']]
# X_test_txfr['Age'] = (X_test_txfr['Age']/100)
# X_test_txfr = X_test_txfr.to_numpy(dtype=float)
# batch_effects_test_txfr = ADNI_DK_clinical[['site2']].to_numpy(dtype=int)


# #%%
# # save the dataframes
# with open(os.path.join(txfer_output_path,'X_test_txfr.pkl'), 'wb') as file:
#     pickle.dump(pd.DataFrame(X_test_txfr), file)
# with open(os.path.join(txfer_output_path, 'Y_test_txfr.pkl'), 'wb') as file:
#     pickle.dump(pd.DataFrame(Y_test_txfr), file) 
# with open(os.path.join(txfer_output_path, 'txbefile.pkl'), 'wb') as file:
#     pickle.dump(pd.DataFrame(batch_effects_test_txfr), file) 

# #%%
# log_dir = txfer_output_path + '/log/'
# os.mkdir(log_dir)

# #%%
# do_qc = True
# model = 'Normal' # 'HBR_HET' 'HBR_HOM'
# method = 'bspline' # 'linear' 'polynomial' 'bspline'
# linear_mu = 'True'
# random_intercept_mu='True'
# linear_sigma   = 'True'
# inscaler = 'standardize' 
# outscaler = 'standardize'
# features = SC_idp_cols()
# #%%



# for idx, measure in enumerate(features):

#     # create an Y_adaptation and Y_test_txfr for each feature
    
#     Y_adaptation_onefeature = Y_adapt[:,idx]
#     with open(os.path.join(txfer_output_path,f'{idx+1}'+'_Y_adaptation_onefeature.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_adaptation_onefeature), file)
 
#     Y_test_txfr_onefeature = Y_test_txfr[:,idx]
#     with open(os.path.join(txfer_output_path,f'{idx+1}'+'_Y_test_txfr_onefeature.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_test_txfr_onefeature), file)
 
           
#     respfile = os.path.join(txfer_output_path,f'{idx+1}'+'_Y_adaptation_onefeature.pkl')
#     covfile = os.path.join(txfer_output_path, 'X_adaptation.pkl')
#     trbefile = os.path.join(txfer_output_path,'adbefile.pkl')
    
#     testrespfile_path = os.path.join(txfer_output_path, f'{idx+1}'+'_Y_test_txfr_onefeature.pkl')
#     testcovfile_path = os.path.join(txfer_output_path,'X_test_txfr.pkl')
#     tsbefile = os.path.join(txfer_output_path, 'txbefile.pkl')
        
#     batch_size = 1
#     memory = '4gb'
#     duration = '24:00:00'
#     inscaler = 'standardize'
#     outscaler = 'standardize'
    
#     #idx = features.index(measure)
    
#     model_path = os.path.join(processing_dir,'SHASHb_1_estimate_scaled_fixed_SC_demented_adults', f'batch_{idx+1}', 'Models')
#     transfer_suffix = '_transfer_'+f'{idx+1}'
    
#     # Run sequentially 
#     yhat, s2, z_scores = ptk.normative.transfer(covfile=covfile, 
#                                                 respfile=respfile,
#                                                 tsbefile=tsbefile, 
#                                                 trbefile=trbefile,
#                                                 model_path = model_path,
#                                                 alg='hbr', 
#                                                 log_path=log_dir, 
#                                                 binary=True,
#                                                 output_path=txfer_output_path, 
#                                                 testcov= testcovfile_path,
#                                                 testresp = testrespfile_path,
#                                                 inscaler = inscaler,
#                                                 outscaler = outscaler,
#                                                 inputsuffix='_estimate',
#                                                 outputsuffix=transfer_suffix, 
#                                                 savemodel=True)
#     with open(os.path.join(txfer_output_path, f'{idx+1}'+'yhat.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(yhat), file)
#     with open(os.path.join(txfer_output_path, f'{idx+1}'+'s2.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(s2), file) 
#     with open(os.path.join(txfer_output_path, f'{idx+1}'+'z_scores.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(z_scores), file) 

# #%% ADNI adaptation predict
# txfer_output_path = os.path.join(processing_dir,'Transfer_SHASHb_1_SC_demented_adults_adapt_predict')
# log_dir = os.path.join(txfer_output_path, "/log")
# os.makedirs(txfer_output_path, exist_ok=True)

# Y_adapt = ADNI_SC_adaptation[features].to_numpy(dtype=float)
# #%%
# for idx, measure in enumerate(features):

#     # create an Y_adaptation and Y_test_txfr for each feature
    
#     Y_adaptation_onefeature = Y_adapt[:,idx]
#     with open(os.path.join(txfer_output_path,f'{idx+1}'+'_Y_adaptation_onefeature.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_adaptation_onefeature), file)
 
#     # Y_test_txfr_onefeature = Y_test_txfr[:,idx]
#     # with open(os.path.join(txfer_output_path,f'{idx+1}'+'_Y_test_txfr_onefeature.pkl'), 'wb') as file:
#     #     pickle.dump(pd.DataFrame(Y_test_txfr_onefeature), file)
 
           
#     respfile = os.path.join(txfer_output_path,f'{idx+1}'+'_Y_adaptation_onefeature.pkl')
#     covfile = os.path.join(txfer_output_path, 'X_adaptation.pkl')
#     trbefile = os.path.join(txfer_output_path,'adbefile.pkl')
    
#     #testrespfile_path = os.path.join(txfer_output_path, f'{idx+1}'+'_Y_test_txfr_onefeature.pkl')
#     #testcovfile_path = os.path.join(txfer_output_path,'X_test_txfr.pkl')
#     #tsbefile = os.path.join(txfer_output_path, 'txbefile.pkl')
        
#     batch_size = 1
#     memory = '4gb'
#     duration = '24:00:00'
#     inscaler = 'standardize'
#     outscaler = 'standardize'
    
#     #idx = features.index(measure)
    
#     model_path = os.path.join(processing_dir,'Transfer_SHASHb_1_SC_demented_adults')
#     transfer_suffix = '_predict_'+f'{idx+1}'
#     inputsuffix = '_transfer_'+f'{idx+1}'
    
#     # Run sequentially 
#     yhat, s2, z_scores = ptk.normative.predict(covfile=covfile, 
#                                                 respfile=respfile,
#                                                 tsbefile=trbefile, 
#                                                 trbefile=trbefile,
#                                                 model_path = model_path,
#                                                 alg='hbr', 
#                                                 log_path=log_dir, 
#                                                 binary=True,
#                                                 output_path=txfer_output_path, 
#                                                 testcov= covfile,
#                                                 testresp = respfile,
#                                                 inscaler = inscaler,
#                                                 outscaler = outscaler,
#                                                 inputsuffix=inputsuffix,
#                                                 outputsuffix=transfer_suffix, 
#                                                 savemodel=True)
#     with open(os.path.join(txfer_output_path, f'{idx+1}'+'yhat.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(yhat), file)
#     with open(os.path.join(txfer_output_path, f'{idx+1}'+'s2.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(s2), file) 
#     with open(os.path.join(txfer_output_path, f'{idx+1}'+'z_scores.pkl'), 'wb') as file:
#         pickle.dump(pd.DataFrame(z_scores), file) 

# #%%
# z = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Transfer_SHASHb_1_SC_demented_adults_adapt_predict/1z_scores.pkl')
# yhat = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Transfer_SHASHb_1_SC_demented_adults_adapt_predict/1yhat.pkl')
# y = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Transfer_SHASHb_1_SC_demented_adults_adapt_predict/1_Y_adaptation_onefeature.pkl')
# s2 = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Transfer_SHASHb_1_SC_demented_adults/1s2.pkl')
# adapt =pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/Transfer_SHASHb_1_SC_demented_adults_adapt_predict/X_adaptation.pkl')
