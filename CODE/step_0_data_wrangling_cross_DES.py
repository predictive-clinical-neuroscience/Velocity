#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:56:53 2024

@author: johbay
"""
#%%
import pandas as pd
import os
import nibabel
import numpy as np
import re
import subprocess
import pcntoolkit as ptk
import seaborn as sns
from matplotlib import pyplot as plt 
import pickle
from utils import DES_idp_cols
from utilities_test import *

#%%
write = True
#%%
# Update desikan
base_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay'
data_out_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DES'

life_tr= pd.read_csv(os.path.join(base_dir, "Lifedata_with_PING/lifespan_big_controls_extended_tr.csv"))
life_te = pd.read_csv(os.path.join(base_dir,"Lifedata_with_PING/lifespan_big_controls_extended_te.csv" ))
life_tr["site"].unique()
life_te["site"].unique()

df = df = pd.concat((life_tr, life_te))

idp_cols = DES_idp_cols()

#assert len(idp_cols) == 68
#%%
life_tr.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
life_te.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
#%%
site_list=life_tr.groupby(by=["site"]).size()

#%% Add OASIS data set
# Male = 1, Female =0
# Diagnosis: 0 = nondemented
OASIS2_dir = '/project_cephfs/3022017.06/OASIS2/freesurfer'
OASIS2_lh = pd.read_csv(os.path.join(OASIS2_dir, 'lh_aparc_a2009s.txt'), sep = "\t")
OASIS2_rh = pd.read_csv(os.path.join(OASIS2_dir, 'rh_aparc_a2009s.txt'), sep = "\t")
OASIS2_euler = pd.read_csv(os.path.join(OASIS2_dir, "euler_number.csv"))
OASIS2_demo = pd.read_csv(os.path.join(OASIS2_dir,"../phenotypes", "oasis_longitudinal_demographics.csv"))

OASIS2_euler =  OASIS2_euler.rename(columns={"Unnamed: 0":"ID"})
OASIS2_demo =  OASIS2_demo.rename(columns={"participant_id":"ID"})

#OASIS2_lh.columns = OASIS2_lh.columns.str.replace("lh", "L", regex=True)
#OASIS2_rh.columns = OASIS2_rh.columns.str.replace("rh", "R", regex=True)

OASIS2_lh = OASIS2_lh.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
OASIS2_rh = OASIS2_rh.rename(columns={"rh.aparc.a2009s.thickness":"ID"})

OASIS2_lh.columns = OASIS2_lh.columns.str.replace("_thickness", "", regex=True)
OASIS2_rh.columns = OASIS2_rh.columns.str.replace("_thickness", "", regex=True)

OASIS2 = OASIS2_lh.merge(OASIS2_rh, on="ID")

# OASIS 2 needs to be coded for repeat measures
site_vec = OASIS2["ID"].str.split(pat="_", expand=True)
OASIS2["site"] = site_vec[0]+ "_" +site_vec[2]

#%% 
OASIS2 = OASIS2.merge(OASIS2_demo, on="ID", how="left")
OASIS2 = OASIS2.merge(OASIS2_euler, on="ID", how="left")

#%%
OASIS2["Mean_Thickness"] = OASIS2[["rh_MeanThickness", "lh_MeanThickness"]].mean(axis=1)
OASIS2.drop(columns=['BrainSegVolNotVent_y','eTIV_y','BrainSegVolNotVent_x','eTIV_x'], inplace=True)

OASIS2.drop(columns=['lh_MeanThickness', 'rh_MeanThickness'], inplace=True)
OASIS2["site_id"] = 800 + OASIS2["Visit_id"]
OASIS2.drop(columns=['matching_id', 'Visit_id'], inplace=True)
OASIS2 = OASIS2.rename(columns={"group":"diagnosis"})
OASIS2["ethnicity"] = 0

OASIS2.sex = OASIS2.sex.str.replace('M', '1')
OASIS2.sex = OASIS2.sex.str.replace('F', '0')
OASIS2.sex = OASIS2.sex.astype(int)


OASIS2["ID_visit"] = OASIS2.ID.str.split(pat="_", expand=True)[2]
OASIS2["ID_visit"] = OASIS2.ID_visit.str.replace("MR", "")

OASIS2["ID_subject"] = OASIS2.ID.str.split(pat="_", expand=True)[0] + "_" + OASIS2.ID.str.split(pat="_", expand=True)[1]

#%%
OASIS2.columns = OASIS2.columns.str.replace("&", "_and_", regex=True)
#idp_cols =DES_idp_cols2()

OASIS2["Median_Thickness"] = OASIS2[idp_cols].median(axis=1)
#%%

OASIS2_demented = OASIS2[(OASIS2["diagnosis"]!="Nondemented")]
OASIS2 = OASIS2[(OASIS2["diagnosis"]=="Nondemented")]
OASIS2.diagnosis = OASIS2.diagnosis.str.replace('Nondemented', '0')
OASIS2.diagnosis = OASIS2.diagnosis.astype(int)

#%%
assert_few_nans(OASIS2)
assert_few_nans(OASIS2_demented)
assert_no_negatives(OASIS2, idp_cols)
assert_no_negatives(OASIS2_demented, idp_cols)
assert_sex_column_is_binary(OASIS2)
assert_sex_column_is_binary(OASIS2_demented)


#%%
if write:
    OASIS2.to_csv(os.path.join(data_out_dir,"csv/OASIS2_cross_DES.csv"),index=False )
    OASIS2.to_pickle(os.path.join(data_out_dir, "OASIS2_cross_DES.pkl"))
    OASIS2_demented.to_csv(os.path.join(data_out_dir,"csv/OASIS2_cross_demented_DES.csv"),index=False )
    OASIS2_demented.to_pickle(os.path.join(data_out_dir, "OASIS2_cross_demented_DES.pkl"))

#%% Update OASIS3
OASIS3_dir = '/project_cephfs/3022017.06/OASIS3/freesurfer'
OASIS3_lh = pd.read_csv(os.path.join(OASIS3_dir, 'lh_aparc_a2009s.txt'), sep = "\t")
OASIS3_rh = pd.read_csv(os.path.join(OASIS3_dir, 'rh_aparc_a2009s.txt'), sep = "\t")
OASIS3_euler = pd.read_csv(os.path.join(OASIS3_dir, "euler_number.csv"))
OASIS3_demo = pd.read_csv(os.path.join(OASIS3_dir,"../docs", "covariates.csv"))

OASIS3_euler =  OASIS3_euler.rename(columns={"Unnamed: 0":"ID"})
OASIS3_demo =  OASIS3_demo.rename(columns={"participant_id":"ID"})

#OASIS3_lh.columns = OASIS3_lh.columns.str.replace("lh", "L", regex=True)
#OASIS3_rh.columns = OASIS3_rh.columns.str.replace("rh", "R", regex=True)

OASIS3_lh = OASIS3_lh.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
OASIS3_rh = OASIS3_rh.rename(columns={"rh.aparc.a2009s.thickness":"ID"})

OASIS3_lh.columns = OASIS3_lh.columns.str.replace("_thickness", "", regex=True)
OASIS3_rh.columns = OASIS3_rh.columns.str.replace("_thickness", "", regex=True)


OASIS3 = OASIS3_lh.merge(OASIS3_rh, on="ID")


idp_cols = DES_idp_cols()
#%%
OASIS3["Mean_Thickness"] = OASIS3[idp_cols].mean(axis=1)
OASIS3["Median_Thickness"] = OASIS3[idp_cols].median(axis=1)


# OASIS needs to be coded for repeat measures
site_vec=OASIS3["ID"].str.split(pat="_", expand=True)
OASIS3["site"]=site_vec[0]+"_"+site_vec[2]


#%% 
OASIS3 = OASIS3.merge(OASIS3_demo, on="ID", how="left")
OASIS3 = OASIS3.merge(OASIS3_euler, on="ID", how="left")
OASIS3 = OASIS3.rename(columns={"gender":"sex", "group":"diagnosis"})
OASIS3.drop(columns=['lh_MeanThickness', 'rh_MeanThickness'], inplace=True)

OASIS3["race_ethnicity"] = 0

OASIS3["ID_subject"] = OASIS3.ID.str.split(pat=".", expand=True)[0]
OASIS3["ID_subject"] = OASIS3.ID_subject.str.split(pat="_", expand=True)[0]

OASIS3 = OASIS3.sort_values("ID")
subjects = OASIS3.ID.str.split(pat="_", expand=True)[0].unique()
OASIS3["ID_visit"]=0

for s,i in enumerate(subjects): 
    res = OASIS3[OASIS3['ID'].str.contains(i)]
    n = len(res.index)
    OASIS3["ID_visit"][OASIS3['ID'].str.contains(i)] = range(n)

OASIS3["ID_visit"]= OASIS3["ID_visit"] +1
#%%
OASIS3_demented = OASIS3[(OASIS3["diagnosis"]!=0)]
OASIS3 = OASIS3[(OASIS3["diagnosis"]==0)]
#OASIS3.diagnosis = OASIS3.diagnosis.str.replace('Nondemented', '0')
#OASIS3.diagnosis = OASIS3.diagnosis.astype(int)

assert_few_nans(OASIS3)
assert_few_nans(OASIS3_demented)
assert_no_negatives(OASIS3, idp_cols)
assert_no_negatives(OASIS3_demented, idp_cols)
assert_sex_column_is_binary(OASIS3)
assert_sex_column_is_binary(OASIS3_demented)


if write:
    OASIS3_demented.to_pickle(os.path.join(data_out_dir, "OASIS3_cross_DES_demented.pkl"))
    OASIS3_demented.to_csv(os.path.join(data_out_dir, "csv/OASIS3_cross_DES_demented.csv"),index=False )

    OASIS3.to_pickle(os.path.join(data_out_dir, "OASIS3_cross_DES.pkl"))
    OASIS3.to_csv(os.path.join(data_out_dir, "csv/OASIS3_cross_DES.csv"),index=False )

#%% Add IMAGEN data set

# get Imagen data set
Imagen_lh = pd.read_csv(os.path.join('/project_cephfs/3022017.06/IMAGEN/freesurfer/lh_aparc_stats_2009s_woFU3.txt'), sep='\t')
Imagen_rh = pd.read_csv(os.path.join('/project_cephfs/3022017.06/IMAGEN/freesurfer/rh_aparc_stats_2009s_woFU3.txt'), sep='\t')

Imagen_FU3_lh = pd.read_csv(os.path.join('/project_cephfs/3022017.06/IMAGEN/freesurfer/lh_aparc_stats_2009s_FU3.txt'), sep='\t')
Imagen_FU3_rh = pd.read_csv(os.path.join('/project_cephfs/3022017.06/IMAGEN/freesurfer/rh_aparc_stats_2009s_FU3.txt'), sep='\t')


#%% Rename - Baseline

# Imaging daya
# Left side

Imagen_lh.columns = Imagen_lh.columns.str.replace("_thickness", "", regex=True)
Imagen_rh.columns = Imagen_rh.columns.str.replace("_thickness", "", regex=True)

Imagen_FU3_lh.columns = Imagen_lh.columns.str.replace("_thickness", "", regex=True)
Imagen_FU3_rh.columns = Imagen_rh.columns.str.replace("_thickness", "", regex=True)


Imagen_lh = Imagen_lh.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
Imagen_rh = Imagen_rh.rename(columns={"rh.aparc.a2009s.thickness":"ID"})

Imagen_FU3_lh = Imagen_FU3_lh.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
Imagen_FU3_rh = Imagen_FU3_rh.rename(columns={"rh.aparc.a2009s.thickness":"ID"})


#%% remove long and split into BL and FU2

Imagen_lh = Imagen_lh[~Imagen_lh["ID"].str.contains("long", na=False)]
Imagen_rh = Imagen_rh[~Imagen_rh["ID"].str.contains("long", na=False)]

Imagen = Imagen_lh.merge(Imagen_rh , on = 'ID')
Imagen_FU3 = Imagen_FU3_lh.merge(Imagen_FU3_rh, on = "ID")

Imagen_BL = Imagen[Imagen["ID"].str.contains(r'.*BL.*', na=False)]
Imagen_FU2 = Imagen[Imagen["ID"].str.contains(r'.*FU2.*', na=False)]

Imagen_BL["ID"]=Imagen_BL.ID.str.split(pat="_", expand=True)[0]
Imagen_FU2["ID"]=Imagen_FU2.ID.str.split(pat="_", expand=True)[0]
Imagen_FU3["ID"]=Imagen_FU3.ID.str.split(pat="_", expand=True)[0]

Imagen_BL["ID_visit"] = 1
Imagen_FU2["ID_visit"] = 2
Imagen_FU3["ID_visit"] = 3

#%%
Imagen_BL.drop(columns=['BrainSegVolNotVent_y','eTIV_y','BrainSegVolNotVent_x','eTIV_x'], inplace=True)
Imagen_FU2.drop(columns=['BrainSegVolNotVent_y','eTIV_y','BrainSegVolNotVent_x','eTIV_x'], inplace=True)
Imagen_FU3.drop(columns=['BrainSegVolNotVent_y','eTIV_y','BrainSegVolNotVent_x','eTIV_x'], inplace=True)


#%% add age and sex

df_IMAGEN_BL_age = pd.read_csv("/project_cephfs/3022017.06/IMAGEN/phenotypes/BL/dawba/IMAGEN_dawba_BL.tsv", sep="\t")
df_IMAGEN_BL_age['age'] = df_IMAGEN_BL_age["age"]/365 
df_IMAGEN_BL_age.index.name
df_IMAGEN = df_IMAGEN_BL_age.reset_index()
df_IMAGEN = df_IMAGEN.rename(columns={"level_0":"ID", "level_1":"age_rounded"})

#%%
df_IMAGEN_BL_sex = pd.read_csv("/project_cephfs/3022017.06/IMAGEN/subjects_v2.7/FU3/participants/IMAGEN_demographics.csv")
df_IMAGEN_BL_sex['sex'] = df_IMAGEN_BL_sex['sex'].str.replace('F','0')
df_IMAGEN_BL_sex['sex'] = df_IMAGEN_BL_sex['sex'].str.replace('M','1')
df_IMAGEN_BL = df_IMAGEN_BL_sex.rename(columns={"PSC2":"ID", "recruitment centre":"site"})

#%% make site vectors
IMAGEN_site_BL = df_IMAGEN_BL[["ID", "site"]]
IMAGEN_site_BL['site_id'] = "NA"

IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "NOTTINGHAM", 'site_id'] = 700
IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "PARIS", 'site_id'] = 701
IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "DRESDEN", 'site_id'] = 702
IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "BERLIN", 'site_id'] = 703
IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "MANNHEIM", 'site_id'] = 704
IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "DUBLIN", 'site_id'] = 705
IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "LONDON", 'site_id'] = 706
IMAGEN_site_BL.loc[IMAGEN_site_BL['site'] == "HAMBURG", 'site_id'] = 707
#%%
# Rename and merge
df_IMAGEN_one = df_IMAGEN[["age", "ID"]]
df_IMAGEN_sex = df_IMAGEN_BL[["sex", "ID"]]

df_IMAGEN_age_sex = df_IMAGEN_one.merge(df_IMAGEN_sex, on = "ID", how="outer")
df_IMAGEN_age_sex_site = df_IMAGEN_age_sex.merge(IMAGEN_site_BL, on ="ID")

Imagen_BL['ID'] = Imagen_BL['ID'].astype(str).str.zfill(12)
df_IMAGEN_age_sex_site['ID'] = df_IMAGEN_age_sex_site['ID'].astype(str).str.zfill(12)

Imagen_BL["ID"] = Imagen_BL["ID"].astype(str)
df_IMAGEN_age_sex_site["ID"] = df_IMAGEN_age_sex_site["ID"].astype(str)

Imagen_BL_full = df_IMAGEN_age_sex_site.merge(Imagen_BL, on = "ID", how = "right")
#%% IMAGEN 2

# Age and sex
df_IMAGEN_FU2 = pd.read_csv("/project_cephfs/3022017.06/IMAGEN/phenotypes/FU2/meta_data/IMAGEN_age_at_mri_acquisition_FU2.csv")
df_IMAGEN_FU2 = df_IMAGEN_FU2.rename(columns={"DTI":"age", "PSC2":"ID"})
df_IMAGEN_FU2_age = df_IMAGEN_FU2[["age", "ID"]]
df_IMAGEN_FU2_age["age"]=df_IMAGEN_FU2_age["age"]/365

#%%
df_IMAGEN_FU2_age_sex = df_IMAGEN_FU2_age.merge(df_IMAGEN_sex, on ="ID", how = "outer")
df_IMAGEN_FU2_age_sex['ID'] = df_IMAGEN_FU2_age_sex['ID'].astype(str).str.zfill(12)

Imagen_FU2["ID"] = Imagen_FU2["ID"].astype(str)
df_IMAGEN_FU2_age_sex["ID"] = df_IMAGEN_FU2_age_sex["ID"].astype(str)

Imagen_FU2_full = df_IMAGEN_FU2_age_sex.merge(Imagen_FU2, on = "ID", how = "right")

#%%
df_IMAGEN_FU3 = pd.read_csv("/project_cephfs/3022017.06/IMAGEN/phenotypes/FU3/dawba/IMAGEN_dawba_FU3.tsv", sep="\t", index_col=False)
df_IMAGEN_FU3 =  df_IMAGEN_FU3.rename(columns={"PSC2":"ID"})
df_IMAGEN_FU3 = df_IMAGEN_FU3[["ID", "age"]]

df_IMAGEN_FU3_sex = df_IMAGEN_FU3.merge(df_IMAGEN_sex, on ="ID", how ="outer")
df_IMAGEN_FU3_sex["ID"] = df_IMAGEN_FU3_sex["ID"].astype(str)

df_IMAGEN_FU3_sex['ID'] = df_IMAGEN_FU3_sex['ID'].astype(str).str.zfill(12)

Imagen_FU3_full = df_IMAGEN_FU3_sex.merge(Imagen_FU3, on ="ID", how = "right")


#%%
#%% Euler number
euler_bl = pd.read_csv('/project_cephfs/3022017.06/IMAGEN/subjects_v2.7/BL/imaging/freesurfer/euler.tsv', sep = "\t")
euler_FU2 = pd.read_csv('/project_cephfs/3022017.06/IMAGEN/subjects_v2.7/FU2/imaging/freesurfer/euler.tsv', sep = "\t")
euler_FU3 = pd.read_csv('/project_cephfs/3022017.06/IMAGEN/subjects_v2.7/FU3/imaging/freesurfer/euler.tsv', sep = "\t")

euler_bl= euler_bl.rename(columns={"orig.nofix": "ID", "rheno":"rh_en", "lheno": "lh_en", "mean":"avg_en"})
euler_FU2 = euler_FU2.rename(columns={"orig.nofix": "ID", "rheno":"rh_en", "lheno": "lh_en", "mean":"avg_en"})
euler_FU3 = euler_FU3.rename(columns={"orig.nofix": "ID", "rheno":"rh_en", "lheno": "lh_en", "mean":"avg_en"})

#%%
# Imagen_bl_all=Imagen_bl_full.merge(euler_bl, on="ID", how="outer")
# df_IMAGEN_FU2_all = df_IMAGEN_FU2_full.merge(euler_FU2, on="ID", how="outer")
# df_IMAGEN_FU3_all = df_IMAGEN_FU3_full.merge(euler_FU3, on="ID", how="outer")

# Imagen_bl_all["ID_visit"] =1
# df_IMAGEN_FU2_all["ID_visit"] =2
# df_IMAGEN_FU3_all["ID_visit"] =3
#%%
IMAGEN=pd.concat([Imagen_BL_full, Imagen_FU2_full, Imagen_FU3_full])

idp_cols = DES_idp_cols()

IMAGEN["Mean_Thickness"] = IMAGEN[["rh_MeanThickness", "lh_MeanThickness"]].mean(axis=1)
IMAGEN["Median_Thickness"] = IMAGEN[idp_cols].median(axis=1)

#%%
IMAGEN.drop(columns=['lh_MeanThickness','rh_MeanThickness'], inplace=True)

#%%
IMAGEN["diagnosis"] = 0 # redo
IMAGEN["race_ethnicity"] = 0
IMAGEN =  IMAGEN.rename(columns={"ID":"ID_subject"})


IMAGEN["sex"] = pd.to_numeric(IMAGEN["sex"], errors="coerce")

assert_few_nans(IMAGEN)
assert_no_negatives(IMAGEN, idp_cols)
assert_sex_column_is_binary(IMAGEN)


if write:
    IMAGEN.to_csv(os.path.join(data_out_dir,"csv/IMAGEN_cross_DES.csv"),index=False )
    IMAGEN.to_pickle(os.path.join(data_out_dir, "IMAGEN_cross_DES.pkl"))
#%% ABCD 
# Male = 1, Female =2

ABCD_race = pd.read_csv('/project_cephfs/3022017.02/phenotypes/ABCD_AnnualRelease5.0/core/abcd-general/abcd_p_demo.csv'
                        , usecols =["race_ethnicity", "eventname", "src_subject_id"])
ABCD = pd.read_csv('/project_cephfs/3022017.02/phenotypes/ABCD_AnnualRelease5.0/core/abcd-general/abcd_y_lt.csv')

ABCD_lh_stats = pd.read_csv('/project_cephfs/3022017.02/freesurfer/lh_aparc.a2009s.stats.txt', sep ="\t")
ABCD_rh_stats = pd.read_csv('/project_cephfs/3022017.02/freesurfer/rh_aparc.a2009s.stats.txt', sep ="\t")
#%%
ABCD_euler = pd.read_csv('/project_cephfs/3022017.02/freesurfer/euler_number.csv')

ABCD_demo = ABCD[["src_subject_id", "eventname", "site_id_l", "interview_age"]]

ABCD_demo["interview_age"] = ABCD_demo["interview_age"]/12
#%%
# Rename eventname

test=ABCD_demo["eventname"].str.split(pat ="_", expand=True)
upper_matrix= pd.concat([test.iloc[:,0].str.title(),test.iloc[:,1].str.title(),
           test.iloc[:,2].str.title(), test.iloc[:,3].str.title(),
           test.iloc[:,4].str.title(), test.iloc[:,5].str.title(),
           test.iloc[:,6].str.title()], axis = 1)


ABCD_demo["eventname_new"]= upper_matrix.fillna('').sum(axis=1)
ABCD_demo['eventname_new'] = ABCD_demo['eventname_new'].str.replace('Base','base')

ABCD_demo["ID"]=ABCD_demo["src_subject_id"] + "_ses-" + ABCD_demo["eventname_new"]
ABCD_demo["ID"]=ABCD_demo.ID.str.replace('NDAR_', 'sub-NDAR')

#%%
test2=ABCD_race["eventname"].str.split(pat ="_", expand=True)
upper_matrix2= pd.concat([test2.iloc[:,0].str.title(),test2.iloc[:,1].str.title(),
           test2.iloc[:,2].str.title(), test2.iloc[:,3].str.title(),
           test2.iloc[:,4].str.title(), test2.iloc[:,5].str.title(),
           test2.iloc[:,6].str.title()], axis = 1)


ABCD_race["eventname_new"]= upper_matrix.fillna('').sum(axis=1)
ABCD_race['eventname_new'] = ABCD_race['eventname_new'].str.replace('Base','base')

ABCD_race["ID"]=ABCD_race["src_subject_id"] + "_ses-" + ABCD_race["eventname_new"]
ABCD_race["ID"]=ABCD_race.ID.str.replace('NDAR_', 'sub-NDAR')
ABCD_race = ABCD_race[["ID", "race_ethnicity"]]
#%%

ABCD_sex = pd.read_csv('/project_cephfs/3022017.02/phenotypes/ABCD_AnnualRelease5.0/core/abcd-general/abcd_p_demo.csv')
ABCD_sex= ABCD_sex[['src_subject_id','eventname','demo_sex_v2']]
test2=ABCD_sex["eventname"].str.split(pat ="_", expand=True)
upper_matrix2= pd.concat([test2.iloc[:,0].str.title(),test2.iloc[:,1].str.title(),
           test2.iloc[:,2].str.title(), test2.iloc[:,3].str.title(),
           test2.iloc[:,4].str.title(), test2.iloc[:,5].str.title(),
           test2.iloc[:,6].str.title()], axis = 1)
ABCD_sex["eventname_new"] = upper_matrix2.fillna('').sum(axis=1)
ABCD_sex['eventname_new'] = ABCD_sex['eventname_new'].str.replace('Base','base')

ABCD_sex["ID"] = ABCD_sex["src_subject_id"] + "_ses-" + ABCD_sex["eventname_new"]
ABCD_sex["ID"] = ABCD_sex.ID.str.replace('NDAR_', 'sub-NDAR')

#%%
#ABCD_lh_stats.columns = ABCD_lh_stats.columns.str.replace("lh", "L", regex=True)
ABCD_lh_stats.columns = ABCD_lh_stats.columns.str.replace("_thickness", "", regex=True)
ABCD_lh_stats=  ABCD_lh_stats.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
ABCD_lh_stats.drop(['eTIV'], axis=1, inplace=True)
ABCD_lh_stats.drop(['BrainSegVolNotVent'], axis=1, inplace=True)


#ABCD_rh_stats.columns = ABCD_rh_stats.columns.str.replace("rh", "R", regex=True)
ABCD_rh_stats.columns = ABCD_rh_stats.columns.str.replace("_thickness", "", regex=True)
ABCD_rh_stats = ABCD_rh_stats.rename(columns={"rh.aparc.a2009s.thickness":"ID"})
ABCD_rh_stats.drop(['eTIV'], axis=1, inplace=True)
ABCD_rh_stats.drop(['BrainSegVolNotVent'], axis=1, inplace=True)


ABCD_stats = ABCD_rh_stats.merge(ABCD_lh_stats,  on="ID", how="outer")
idp_cols = DES_idp_cols()

ABCD_stats["Mean_Thickness"] = ABCD_stats[["rh_MeanThickness", "lh_MeanThickness"]].mean(axis=1)

ABCD_stats["Median_Thickness"] = ABCD_stats[idp_cols].median(axis=1)

#%% Merge
ABCD_full = ABCD_stats.merge(ABCD_demo, on="ID")
ABCD_full = ABCD_full.merge(ABCD_sex, on = "ID")
ABCD_full = ABCD_full.merge(ABCD_race, on = "ID",how="left")

# refill sex and ethnicity based on previous entries
ref = ABCD_full.groupby('src_subject_id_y')['race_ethnicity'].first()
ABCD_full['race_ethnicity'] = ABCD_full['race_ethnicity'].fillna(ABCD_full['src_subject_id_y'].map(ref))

ref = ABCD_full.groupby('src_subject_id_y')['demo_sex_v2'].first()
ABCD_full['demo_sex_v2'] = ABCD_full['demo_sex_v2'].fillna(ABCD_full['src_subject_id_y'].map(ref))


ABCD_full = ABCD_full.drop([ 'eventname_new_x',
                'eventname_y', 'eventname_new_y', 'src_subject_id_y'], axis =1 )


ABCD_full.rename(columns={"site_id_l": "site", "interview_age": "age", 
                          'eventname_x': "ID_visit", "demo_sex_v2":"sex",
                          "src_subject_id_x":"ID_subject"}, inplace=True)

ABCD_full = ABCD_full.replace({'ID_visit':{'4_year_follow_up_y_arm_1':'4', 'baseline_year_1_arm_1':'1', 
                                   '2_year_follow_up_y_arm_1':'2', '3_year_follow_up_y_arm_1':'3'}}, regex = True)


# recode sex

ABCD_full.sex.replace(2, 0, inplace=True)
ABCD_full["site_id"] = "540" + ABCD_full["site"].str.replace("site", "", regex=True)
ABCD_full["site"] = "abcd_" + ABCD_full["site"]

ABCD_full.columns = ABCD_full.columns.str.replace("_thickness", "", regex=True)



#check types of columns
assert ABCD_full.sex.dtype == float
assert ABCD_full.age.dtype == float

ABCD_full = ABCD_full[ABCD_full["sex"].isin([0, 1])]
assert_few_nans(ABCD_full)
assert_no_negatives(ABCD_full, idp_cols)
assert_sex_column_is_binary(ABCD_full)



if write:
    ABCD_full.to_pickle(os.path.join(data_out_dir,"ABCD_cross_DES.pkl"))
    ABCD_full.to_csv(os.path.join(data_out_dir,"csv/ABCD_cross_DES.csv"))
#%% UKB
# Sex encoding: Female = 0, Male =1
# Race encoding: 
    
UKB_race = pd.read_csv("/project_cephfs/3022017.05/phenotypes/current/02_genetics.csv", usecols = ["eid","21000-0.0"])
UKB_age1 = pd.read_csv("/project_cephfs/3022017.05/phenotypes/age_sex_site_ukb35187.csv")
UKB_age2 = pd.read_csv("/project_cephfs/3022017.05/phenotypes/age_sex_site_ukb42006.csv")

UKB_age1.rename(columns={"eid":"ID","31-0.0":"sex", "54-2.0":"site", "21003-2.0":"age"}, inplace = True)

UKB_age2.rename(columns={"eid":"ID_subject","31-0.0":"sex", "54-2.0":"site","54-3.0":"site2", "21003-2.0":"age1","21003-3.0":"age2"}, inplace = True)
UKB_age2["ID_subject"] = UKB_age2["ID_subject"].astype(str)


UKB_lh = pd.read_csv(os.path.join(base_dir,"DATA_Longitudinal_with_PING","lh_aparc_stats_all_a2009s_UKB.txt"), sep = "\t")
UKB_rh = pd.read_csv(os.path.join(base_dir,"DATA_Longitudinal_with_PING","rh_aparc_stats_all_a2009s_UKB.txt"), sep = "\t")

#UKB_lh.columns = UKB_lh.columns.str.replace("lh", "L", regex=True)
#UKB_rh.columns = UKB_rh.columns.str.replace("rh", "R", regex=True)

UKB_lh = UKB_lh.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
UKB_rh = UKB_rh.rename(columns={"rh.aparc.a2009s.thickness":"ID"})


UKB_lh.drop_duplicates(inplace=True)
UKB_rh.drop_duplicates(inplace=True)


UKB_lh.drop(columns=['lh_MeanThickness_thickness'], inplace=True)
UKB_rh.drop(columns=['rh_MeanThickness_thickness'], inplace=True)

column_to_delete_from = "BrainSegVolNotVent"  

# Get the index of the specified column
col_index = UKB_rh.columns.get_loc(column_to_delete_from)

# Drop all columns from that column onward
UKB_rh = UKB_rh.iloc[:, :col_index] 
UKB_lh = UKB_lh.iloc[:, :col_index] 

UKB_lh_rh = UKB_lh.merge(UKB_rh, on = "ID")

threshold = 0.1 * UKB_lh_rh.shape[1]  # 10% of total columns

# Count the number of zeros in each row
zero_counts = (UKB_lh_rh == 0).sum(axis=1)

# Keep only rows where the number of zeros is <= threshold
UKB_lh_rh = UKB_lh_rh[zero_counts <= threshold]

#%% convert race
assert UKB_race["21000-0.0"].dtype == 'float64'

UKB_race["21000-0.0"].unique()
UKB_race["race_ethnicity"] = UKB_race["21000-0.0"]
UKB_race.replace({'race_ethinicty': {1001 : 1, 1002 : 1, 1003 : 1, 2000 : 2, 
                           2001 : 2, 2003 : 2, 2004 : 2, 2002 : 2, 
                           3000 : 3, 3001 : 3, 3002 : 3, 3003 : 3, 3004 : 3,
                           4000 : 4, 4001 : 4, 4002 : 4, 4003 : 4,
                           5000 : 5, 6000 : 6}}, inplace =True)
UKB_race["race_ethnicity"].unique()
UKB_race.rename(columns={"eid":"ID"}, inplace = True)
UKB_race.drop(columns={"21000-0.0"}, inplace=True)


#%% 

# create a subject ID
UKB_lh_rh  = UKB_lh_rh[~UKB_lh_rh["ID"].str.contains("long", na=False)]
UKB_lh_rh["ID_visit"] = UKB_lh_rh["ID"].str.split(pat ="_", expand=True)[1]
UKB_lh_rh = UKB_lh_rh.fillna(value="1")
UKB_lh_rh["ID_visit"] = UKB_lh_rh["ID_visit"].str.replace('scan2', '2')
UKB_lh_rh["ID_visit"] = UKB_lh_rh["ID_visit"].astype(int)
UKB_lh_rh["ID_subject"] = UKB_lh_rh["ID"].str.split(pat ="_", expand=True)[0]

UKB_lh_rh["ID_subject"] = UKB_lh_rh["ID_subject"].astype(int)

UKB_lh_rh.columns = UKB_lh_rh.columns.str.replace("_thickness", "", regex=True)


#UKB = UKB_lh_rh.merge(UKB_race, left_on ="ID_subject", right_on="ID")
#UKB = UKB.merge(UKB_age1, on = "ID_subject")
#UKB.sex = UKB.sex.astype(float)

idp_cols = DES_idp_cols()

UKB_lh_rh.columns = UKB_lh_rh.columns.str.replace("&", "_and_", regex=True)

UKB_lh_rh["Mean_Thickness"] = UKB_lh_rh[idp_cols].mean(axis=1)
UKB_lh_rh["Median_Thickness"] = UKB_lh_rh[idp_cols].mean(axis=1)

# create variables for age
UKB_age=UKB_age2.drop(columns="site2")

UKB_age_1 = UKB_age[["ID_subject", "site", "sex","age1"]]
UKB_age_1["ID_visit"]=1
UKB_age_1 = UKB_age_1.dropna(how="any")

UKB_age_2 = UKB_age[["ID_subject", "site", "sex","age2"]]
UKB_age_2["ID_visit"]=2
UKB_age_2 = UKB_age_2.dropna(how="any")

UKB_age_1.rename(columns={"age1":"age"}, inplace=True)
UKB_age_2.rename(columns={"age2":"age"}, inplace=True)

UKB_age = [UKB_age_1, UKB_age_2]
UKB_age = pd.concat(UKB_age)


UKB_lh_rh[["ID_subject", "ID_visit"]] =  UKB_lh_rh[["ID_subject", "ID_visit"]].astype(int)
UKB_age[["ID_subject", "ID_visit"]] =  UKB_age[["ID_subject", "ID_visit"]].astype(int)


#%%
UKB = UKB_lh_rh.merge(UKB_age, on = ["ID_subject", "ID_visit"])
UKB["site_id"] = UKB["site"]

assert UKB.sex.dtype == float
assert UKB.age.dtype == float
#assert UKB.race_ethnicity.dtype == float


assert_few_nans(UKB)
assert_no_negatives(UKB, idp_cols)
assert_sex_column_is_binary(UKB)


if write:
    UKB.to_pickle(os.path.join(data_out_dir, "UKB_cross_DES.pkl"))
    UKB.to_csv(os.path.join(data_out_dir, "csv/UKB_cross_DES.csv"))


#%% HCP Babies
# Sex encoding: 0: Female, 1: Male
# Race encoding: White', 'Asian', 'More than one race','Black or African American'

HCP_babies_pass = pd.read_csv("/project_cephfs/3022017.06/HCP_Baby/freesurfer/quality_HCP_Babies.csv",
                              sep = ";", usecols = ["ID","pass"])

HCP_Babies_sex = pd.read_csv("/project_cephfs/3022017.06/HCP_Baby/phenotypes/image03.txt", sep = "\t"
                              ,usecols = ["subjectkey", "sex", "src_subject_id", "interview_date"])

HCP_Babies_age  = pd.read_csv("/project_cephfs/3022017.06/HCP_Baby/phenotypes/image03.txt", sep = "\t"
                            ,usecols = ["subjectkey","interview_age", "src_subject_id", "interview_date"])

#%%

HCP_Babies_sex = HCP_Babies_sex.iloc[1:]

HCP_Babies_sex.sex = HCP_Babies_sex.sex.str.replace('M', '1')
HCP_Babies_sex.sex = HCP_Babies_sex.sex.str.replace('F', '0')
HCP_Babies_sex.rename(columns={"subjectkey":"ID"}, inplace = True)

#%%

HCP_Babies_age = HCP_Babies_age.iloc[1:]
HCP_Babies_age.drop_duplicates(inplace=True)

test= HCP_Babies_age["interview_date"].str.split(pat="/", expand=True)
HCP_Babies_age["scan_date"] = test[2]+test[0]+ test[1]
HCP_Babies_age["ID_visit"] = HCP_Babies_age["src_subject_id"].astype(str) +"_"+ HCP_Babies_age["scan_date"].astype(str)

HCP_Babies_age.rename(columns={"interview_age":"age"}, inplace = True)

HCP_Babies_age['age'] = HCP_Babies_age['age'].astype(float)
HCP_Babies_age['age'] = HCP_Babies_age['age']/12

HCP_Babies_age = HCP_Babies_age[["age", "src_subject_id", "ID_visit"]]
HCP_Babies_age.drop_duplicates(inplace=True)
#%%

HCP_Babies_rh = pd.read_csv("/project_cephfs/3022017.06/HCP_Baby/freesurfer/rh_aparc_stats_HCP_Babies_a2009s.txt", sep = "\t")
HCP_Babies_lh = pd.read_csv("/project_cephfs/3022017.06/HCP_Baby/freesurfer/lh_aparc_stats_HCP_Babies_a2009s.txt", sep = "\t")

HCP_Babies_lh= HCP_Babies_lh.drop_duplicates()
HCP_Babies_rh= HCP_Babies_rh.drop_duplicates()

HCP_babies_ids = pd.read_csv('/project_cephfs/3022017.06/HCP_Baby/phenotypes/ndar_subject01.txt', sep = "\t",
                             usecols= ["src_subject_id", "race", "ethnic_group"])

HCP_babies_ids = HCP_babies_ids.iloc[1:]

HCP_babies_ids.race.unique()

#HCP_Babies_lh.columns = HCP_Babies_lh.columns.str.replace("lh", "L", regex=True)
#HCP_Babies_rh.columns = HCP_Babies_rh.columns.str.replace("rh", "R", regex=True)

HCP_Babies_lh = HCP_Babies_lh.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
HCP_Babies_rh = HCP_Babies_rh.rename(columns={"rh.aparc.a2009s.thickness":"ID"})

HCP_Babies = HCP_Babies_lh.merge(HCP_Babies_rh, on = "ID")
#idp_cols = ut.return_idp_cols()


HCP_Babies.drop(columns=['lh_MeanThickness_thickness'], inplace=True)
HCP_Babies.drop(columns=['rh_MeanThickness_thickness'], inplace=True)

HCP_Babies.columns = HCP_Babies.columns.str.replace("_thickness", "", regex=True)

HCP_Babies["Mean_Thickness"]= HCP_Babies[idp_cols].mean(axis=1)
HCP_Babies["Median_Thickness"]= HCP_Babies[idp_cols].median(axis=1)


#%%
HCP_Babies = HCP_Babies.merge(HCP_babies_pass, on = "ID")

#%%
HCP_Babies["src_subject_id"] = HCP_Babies.ID.str.split(pat = "_", expand=True)[0]
HCP_Babies["src_subject_id"] = HCP_Babies["src_subject_id"].replace("MNBCP", "", regex=True)
HCP_Babies["src_subject_id"] = HCP_Babies["src_subject_id"].replace("NCBCP", "", regex=True)
HCP_Babies["src_subject_id"] = HCP_Babies["src_subject_id"].replace("MNBC", "", regex=True)


test2=HCP_Babies["ID"].str.split(pat="-", expand=True)[3]
HCP_Babies["scan_date"]=test2.str.split(pat=".", expand=True)[0]
HCP_Babies["ID_visit"] = HCP_Babies["src_subject_id"] + "_"+ HCP_Babies["scan_date"]


#%%
HCP_Babies_age = HCP_Babies_age[["ID_visit", "age"]]
HCP_Babies_age.drop_duplicates(inplace=True)
HCP_Babies = HCP_Babies.merge(HCP_Babies_age,  on="ID_visit", how ="left")
HCP_babies_ids = HCP_babies_ids.drop_duplicates()
#%%
HCP_Babies= HCP_Babies.merge(HCP_babies_ids, on="src_subject_id", how = "left")

#%%
HCP_Babies["site"]= "U"+HCP_Babies["ID"].str[:2]

#%%
HCP_Babies_sex = HCP_Babies_sex[["src_subject_id", "sex"]]
HCP_Babies_sex.drop_duplicates(inplace=True)
HCP_Babies = HCP_Babies.merge(HCP_Babies_sex, on="src_subject_id", how = "left")

#%%
HCP_Babies = HCP_Babies[HCP_Babies["pass"]==1]
HCP_Babies = HCP_Babies.drop_duplicates()

#%%

ref = HCP_Babies.groupby('src_subject_id')['sex'].first()
HCP_Babies['sex'] = HCP_Babies['sex'].fillna(HCP_Babies['src_subject_id'].map(ref))
HCP_Babies.drop(columns=["race", "ethnic_group"], inplace = True)

#%%
HCP_Babies = HCP_Babies.dropna(how = 'any')
HCP_Babies = HCP_Babies[~HCP_Babies["ID"].str.contains('long')]
HCP_Babies["site_id"] = 9001

HCP_Babies["ID_visit"] = HCP_Babies["ID"].str.split(pat="-", expand=True)[0]
HCP_Babies["ID_visit"] = HCP_Babies["ID_visit"].str.split(pat="_", expand=True)[1]
HCP_Babies["ID_visit"] = HCP_Babies["ID_visit"].str.replace("v0", "")
HCP_Babies["ID_visit"] = HCP_Babies["ID_visit"].astype(int)

HCP_Babies = HCP_Babies.rename(columns={"src_subject_id":"ID_subject"})

if write:
    HCP_Babies.to_pickle(os.path.join(data_out_dir, "HCP_Babies_cross_DES.pkl"))

#%% NIH Babies

NIH_Babies_lh  = pd.read_csv("/project_cephfs/3022017.06/NIH_pedMRI/freesurfer/lh_aparc_stats_a2009s.txt", sep = "\t")
NIH_Babies_rh  = pd.read_csv("/project_cephfs/3022017.06/NIH_pedMRI/freesurfer/rh_aparc_stats_a2009s.txt", sep = "\t")

#NIH_Babies_lh.columns = NIH_Babies_lh.columns.str.replace("lh", "L", regex=True)
#NIH_Babies_rh.columns = NIH_Babies_rh.columns.str.replace("rh", "R", regex=True)

NIH_Babies_lh = NIH_Babies_lh.rename(columns={"lh.aparc.a2009s.thickness":"ID"})
NIH_Babies_rh = NIH_Babies_rh.rename(columns={"rh.aparc.a2009s.thickness":"ID"})

NIH_Babies_lh.drop(columns=['BrainSegVolNotVent','eTIV'], inplace=True)
NIH_Babies_rh.drop(columns=['BrainSegVolNotVent','eTIV'], inplace=True)

NIH_Babies_lh.drop(columns=['lh_MeanThickness_thickness'], inplace=True)
NIH_Babies_rh.drop(columns=['rh_MeanThickness_thickness'], inplace=True)

idp_cols = DES_idp_cols()

NIH_Babies = NIH_Babies_lh.merge(NIH_Babies_rh, on = "ID")

NIH_Babies.columns = NIH_Babies.columns.str.replace("_thickness", "", regex=True)

NIH_Babies["Mean_Thickness"]= NIH_Babies[idp_cols].mean(axis=1)
NIH_Babies["Median_Thickness"]= NIH_Babies[idp_cols].median(axis=1)

NIH_Babies = NIH_Babies[~NIH_Babies["ID"].astype(str).str.contains("long", case=False, na=False)]


#%%

NIH_Babies["ID_subject"] = NIH_Babies.ID.str.split(pat="_", expand=True)[0]
NIH_Babies["ID_visit"] = NIH_Babies.ID.str.split(pat="_", expand=True)[1]

#%%


NIH_Babies_demo = pd.read_csv('/project_cephfs/3022017.06/NIH_pedMRI/phenotypes/peds_demographics01.txt', sep = "\t",
                              usecols = ["subject_gender", "child_race_", "age_years_dov_to_dob",
                                         "timepoint_label", "site_id", "src_subject_id"] )

NIH_Babies_demo = NIH_Babies_demo.iloc[1:]

NIH_Babies_demo.rename(columns={"src_subject_id":"ID_subject", "timepoint_label":"ID_visit",
                                "subject_gender": "sex", "child_race_":"race", "age_years_dov_to_dob":"age"},
                       inplace=True)

#%%
NIH_Babies_demo["race"].unique()

NIH_Babies_demo["sex"]=NIH_Babies_demo["sex"].str.replace('Male', '1')
NIH_Babies_demo["sex"]=NIH_Babies_demo["sex"].str.replace('Female', '0')

#NIH_Babies_demo["race"]=NIH_Babies_demo["race"].str.replace('/Not Provided', '')

NIH_Babies_demo["site"] = "NIH_Babies_" + NIH_Babies_demo["site_id"] 
NIH_Babies_demo["site_id"] = "77" + NIH_Babies_demo["site_id"] 

NIH_Babies_demo["ID_visit"]=NIH_Babies_demo["ID_visit"].str.replace('V', '')

NIH_Babies_demo["ID_visit"] = NIH_Babies_demo["ID_visit"].astype(str)
NIH_Babies["ID_visit"] = NIH_Babies["ID_visit"].astype(str)
NIH_Babies["ID_visit"]=NIH_Babies["ID_visit"].str.replace('v', '')

NIH_Babies_demo["ID_subject"] = NIH_Babies_demo["ID_subject"].astype(str)
NIH_Babies["ID_subject"] = NIH_Babies["ID_subject"].astype(str)


#%% merge
NIH_Babies = pd.merge(
    left=NIH_Babies, 
    right=NIH_Babies_demo,
    how='left',
    on=['ID_subject', 'ID_visit']
)

#fill nans in race

race_ref = NIH_Babies.groupby('ID_subject')['race'].first()
NIH_Babies['race'] = NIH_Babies['race'].fillna(NIH_Babies['ID_subject'].map(race_ref))

HCP_Babies.ID_visit = HCP_Babies.ID_visit.astype(int)

if write:
    NIH_Babies.to_pickle(os.path.join(data_out_dir, "NIH_Babies_cross_DES.pkl"))


#%% Compare dataframes
OASIS2_cols = OASIS2.columns.to_list()
OASIS3_cols = OASIS3.columns.to_list()
ABCD_cols = ABCD_full.columns.to_list()
IMAGEN_cols = IMAGEN.columns.to_list()
life_tr_cols = life_tr.columns.to_list()
res = [x for x in OASIS2_cols + life_tr_cols if x not in life_tr_cols or x not in OASIS2_cols]

  