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
from utils import SC_idp_cols
from utilities_test import *


#%%
def test_sex(df, column_name="sex"):
    valid_values = {0, 1}
    unique_values = set(df[column_name].dropna().unique())
    assert unique_values <= valid_values, f"{column_name} contains values other than 0 and 1: {unique_values}"


#%%
idp_cols = SC_idp_cols()
write = True
#%%
# Update desikan
base_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay'
data_out_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC'

#%%
life_tr= pd.read_csv(os.path.join(base_dir, "Lifedata_with_PING/lifespan_big_controls_sc_extended_tr.csv"))
life_te = pd.read_csv(os.path.join(base_dir,"Lifedata_with_PING/lifespan_big_controls_sc_extended_te.csv" ))
life_tr["site"].unique()
life_te["site"].unique()


df = df = pd.concat((life_tr, life_te))

#%%
#life_1 = pd.read_csv(os.path.join(base_dir, "Lifedata_with_PING/DK_extended_lifespan_baby_big_ct_te.csv"))
#life_2 = pd.read_csv(os.path.join(base_dir,"Lifedata_with_PING/DK_extended_lifespan_big_ct_te.csv"))
#life_3 = pd.read_csv(os.path.join(base_dir, "Lifedata_with_PING/lifespan_big_controls_extended_te.csv"))
#%%
life_tr.rename(columns={'Unnamed: 0':'ID', 'Left-Thalamus-Proper':'Left-Thalamus', 'Right-Thalamus-Proper':'Right-Thalamus'}, inplace=True)
life_te.rename(columns={'Unnamed: 0':'ID', 'Left-Thalamus-Proper':'Left-Thalamus', 'Right-Thalamus-Proper':'Right-Thalamus'}, inplace=True)
#%%
site_list=life_tr.groupby(by=["site"]).size()

#%% Add OASIS data set
# Male = 1, Female =0
# Diagnosis: 0 = nondemented
OASIS2_dir = '/project_cephfs/3022017.06/OASIS2/freesurfer'
OASIS2 = pd.read_csv(os.path.join(OASIS2_dir, 'aseg_stats_all.txt'), sep = "\t")

OASIS2["Left-Thalamus"] = OASIS2["Left-Thalamus"] +OASIS2["Left-Thalamus-Proper"]
OASIS2["Right-Thalamus"] = OASIS2["Right-Thalamus"] +OASIS2["Right-Thalamus-Proper"]


#%% create a corss sectional and a longitudinal data frame

OASIS2_demo = pd.read_csv(os.path.join(OASIS2_dir,"../phenotypes", "oasis_longitudinal_demographics.csv"))
OASIS2_demo =  OASIS2_demo.rename(columns={"participant_id":"ID", "matching_ID":"subject_ID"})
OASIS2 = OASIS2.rename(columns={"Measure:volume":"ID_long"})

# OASIS 2 needs to be coded for repeat measures
site_vec = OASIS2["ID_long"].str.split(pat="_", expand=True)
OASIS2["site"] = site_vec[0]+ "_" +site_vec[2]

#%% drop the  templates
OASIS2 = OASIS2[~OASIS2['site'].str.endswith('_long', na=False)] 

OASIS2["subject_ID"] = OASIS2["ID_long"].str.split(pat=".", expand=True)[0]
OASIS2["processing_type"] = OASIS2["site"].str.split(pat=".", expand=True)[1]
OASIS2['processing_type'] = OASIS2['processing_type'].fillna('cross')
#%% 
OASIS2 = OASIS2.merge(OASIS2_demo, right_on="ID", left_on="subject_ID")
#OASIS2 = OASIS2.merge(OASIS2_euler, on="ID", how="left")

#%%
OASIS2["site_id"] = 800 
#OASIS2.drop(columns=['matching_id', 'Visit_id'], inplace=True)
OASIS2 = OASIS2.rename(columns={"group":"diagnosis"})
#OASIS2["ethnicity"] = 0

OASIS2.sex = OASIS2.sex.str.replace('M', '1')
OASIS2.sex = OASIS2.sex.str.replace('F', '0')
OASIS2.sex = OASIS2.sex.astype(int)
#%%
OASIS2.drop(columns=['subject_ID'], inplace=True)
OASIS2 = OASIS2.rename(columns={"matching_id":"ID_subject", "Visit_id":"ID_visit"})

OASIS2_demented = OASIS2[(OASIS2["diagnosis"]!="Nondemented")]
OASIS2 = OASIS2[(OASIS2["diagnosis"]=="Nondemented")]
OASIS2.diagnosis = OASIS2.diagnosis.str.replace('Nondemented', '0')
OASIS2_demented.diagnosis = OASIS2_demented.diagnosis.str.replace('Converted', '1')
OASIS2_demented.diagnosis = OASIS2_demented.diagnosis.str.replace('Demented', '2')
OASIS2_demented.diagnosis = OASIS2_demented.diagnosis.astype(int)
OASIS2.diagnosis = OASIS2.diagnosis.astype(int)

OASIS2_cross = OASIS2[OASIS2["processing_type"]=="cross"]
OASIS2_long = OASIS2[OASIS2["processing_type"]=="long"]

OASIS2_cross_demented = OASIS2_demented[OASIS2_demented["processing_type"]=="cross"]
OASIS2_long_demented = OASIS2_demented[OASIS2_demented["processing_type"]=="long"]

#%% tests
test_sex(OASIS2_long_demented)
test_sex(OASIS2_cross_demented)

test_sex(OASIS2_long)
test_sex(OASIS2_cross)

assert_few_nans(OASIS2_cross_demented)
assert_few_nans(OASIS2_long_demented)
assert_few_nans(OASIS2_cross)
assert_few_nans(OASIS2_long)

assert_no_negatives(OASIS2_cross_demented, idp_cols)
assert_no_negatives(OASIS2_long_demented, idp_cols)
assert_no_negatives(OASIS2_cross, idp_cols)
assert_no_negatives(OASIS2_long, idp_cols)

if write:
    OASIS2_cross.to_csv(os.path.join(data_out_dir,"csv/OASIS2_cross_SC_.csv"),index=False )
    OASIS2_cross.to_pickle(os.path.join(data_out_dir, "OASIS2_cross_SC.pkl"))
    OASIS2_cross_demented.to_csv(os.path.join(data_out_dir,"csv/OASIS2_cross_SC_demented.csv"),index=False )
    OASIS2_cross_demented.to_pickle(os.path.join(data_out_dir,"OASIS2_cross_SC_demented.pkl"))
    OASIS2_long.to_csv(os.path.join(data_out_dir,"csv/OASIS2_long_SC.csv"),index=False )
    OASIS2_long.to_pickle(os.path.join(data_out_dir, "OASIS2_long_SC.pkl"))
    OASIS2_long_demented.to_csv(os.path.join(data_out_dir,"csv/OASIS2_long_SC_demented.csv"),index=False )
    OASIS2_long_demented.to_pickle(os.path.join(data_out_dir,"OASIS2_long_SC_demented.pkl"))
#
#%% Update OASIS3
OASIS3_dir = '/project_cephfs/3022017.06/OASIS3/freesurfer'
OASIS3 = pd.read_csv(os.path.join(OASIS3_dir, 'aseg_stats_all.txt'), sep = "\t")

OASIS3["Left-Thalamus"] = OASIS3["Left-Thalamus"] +OASIS3["Left-Thalamus-Proper"]
OASIS3["Right-Thalamus"] = OASIS3["Right-Thalamus"] +OASIS3["Right-Thalamus-Proper"]

#OASIS3_euler = pd.read_csv(os.path.join(OASIS3_dir, "euler_number.csv"))
OASIS3_demo = pd.read_csv(os.path.join(OASIS3_dir,"../docs", "covariates.csv"))

#OASIS3_euler =  OASIS3_euler.rename(columns={"Unnamed: 0":"ID"})
OASIS3_demo =  OASIS3_demo.rename(columns={"participant_id":"ID"})
OASIS3 = OASIS3.rename(columns={"Measure:volume":"ID_long"})


# OASIS needs to be coded for repeat measures
site_vec=OASIS3["ID_long"].str.split(pat="_", expand=True)
OASIS3["site"]=site_vec[0]+"_"+site_vec[2]

OASIS3["ID"] = OASIS3["ID_long"].str.split(pat="_", expand=True)[0] + "_" + \
    OASIS3["ID_long"].str.split(pat="_", expand=True)[1] + "_" + OASIS3["ID_long"].str.split(pat="_", expand=True)[2]

OASIS3['ID'] = OASIS3['ID_long'].apply(
    lambda x: x.split('.long.')[0] if '.long.' in x else x
)


#%% remove templates
OASIS3 = OASIS3[~OASIS3['site'].str.endswith('_long', na=False)] 

#%% 
OASIS3 = OASIS3.merge(OASIS3_demo, on="ID", how="left")
#OASIS3 = OASIS3.merge(OASIS3_euler, on="ID", how="left")
OASIS3 = OASIS3.rename(columns={"gender":"sex", "group":"diagnosis"})

OASIS3["processing_type"] = OASIS3["site"].str.split(pat=".", expand=True)[1]
OASIS3['processing_type'] = OASIS3['processing_type'].fillna('cross')

OASIS3["race_ethnicity"] = 0

OASIS3["ID_subject"] = OASIS3.ID.str.split(pat=".", expand=True)[0]
OASIS3["ID_subject"] = OASIS3.ID_subject.str.split(pat="_", expand=True)[0]

OASIS3_cross = OASIS3[OASIS3["processing_type"] =="cross"]
OASIS3_long = OASIS3[OASIS3["processing_type"] =="long"]

#%% cross
OASIS3_cross = OASIS3_cross.sort_values("ID")
subjects = OASIS3_cross.ID.str.split(pat="_", expand=True)[0].unique()
OASIS3_cross["ID_visit"]=0


for s,i in enumerate(subjects): 
    res = OASIS3_cross[OASIS3_cross['ID'].str.contains(i)]
    n = len(res.index)
    OASIS3_cross["ID_visit"][OASIS3_cross['ID'].str.contains(i)] = range(n)

OASIS3_cross["ID_visit"]= OASIS3_cross["ID_visit"] +1

#%% long
OASIS3_long = OASIS3_long.sort_values("ID")
subjects = OASIS3_long.ID.str.split(pat="_", expand=True)[0].unique()
OASIS3_long["ID_visit"]=0

for s,i in enumerate(subjects): 
    res = OASIS3_long[OASIS3_long['ID'].str.contains(i)]
    n = len(res.index)
    OASIS3_long["ID_visit"][OASIS3_long['ID'].str.contains(i)] = range(n)

OASIS3_long["ID_visit"]= OASIS3_long["ID_visit"] +1

#%%s
OASIS3_long_demented = OASIS3_long[OASIS3_long["diagnosis"]!=0]
OASIS3_cross_demented = OASIS3_cross[OASIS3_cross["diagnosis"]!=0]

OASIS3_long = OASIS3_long[OASIS3_long["diagnosis"]==0]
OASIS3_cross = OASIS3_cross[OASIS3_cross["diagnosis"]==0]

test_sex(OASIS3_long_demented)
test_sex(OASIS3_cross_demented)

test_sex(OASIS3_long)
test_sex(OASIS3_cross)

if write:
    OASIS3_cross.to_pickle(os.path.join(data_out_dir, "OASIS3_cross_SC.pkl"))
    OASIS3_cross.to_csv(os.path.join(data_out_dir, "csv/OASIS3_cross_SC.csv"),index=False )
    OASIS3_long.to_pickle(os.path.join(data_out_dir, "OASIS3_long_SC.pkl"))
    OASIS3_long.to_csv(os.path.join(data_out_dir, "csv/OASIS3_long_SC.csv"),index=False )
    OASIS3_cross_demented.to_pickle(os.path.join(data_out_dir, "OASIS3_cross_SC_demented.pkl"))
    OASIS3_cross_demented.to_csv(os.path.join(data_out_dir, "csv/OASIS3_cross_SC_demented.csv"),index=False )
    OASIS3_long_demented.to_pickle(os.path.join(data_out_dir, "OASIS3_long_SC_demented.pkl"))
    OASIS3_long_demented.to_csv(os.path.join(data_out_dir, "csv/OASIS3_long_SC_demented.csv"),index=False )

#%% Add IMAGEN data set

# get Imagen data set

Imagen = pd.read_csv(os.path.join('/project_cephfs/3022017.06/IMAGEN/freesurfer/aseg_stats_all.txt'),  sep='\t')

#%% Split data set into BL, FU2 and FU3
Imagen = Imagen.rename(columns={"Measure:volume":"ID_long"})
Imagen["ID_long"] = Imagen["ID_long"].astype(str)

Imagen["Left-Thalamus"] = Imagen["Left-Thalamus"] +Imagen["Left-Thalamus-Proper"]
Imagen["Right-Thalamus"] = Imagen["Right-Thalamus"] +Imagen["Right-Thalamus-Proper"]


Imagen_BL = Imagen[Imagen["ID_long"].str.contains(r'.*BL.*', na=False)]
Imagen_FU2 = Imagen[Imagen["ID_long"].str.contains(r'.*FU2.*', na=False)]
Imagen_FU3 = Imagen[Imagen["ID_long"].str.contains(r'.*FU3.*', na=False)]

Imagen_BL["ID"]=Imagen_BL.ID_long.str.split(pat="_", expand=True)[0]
Imagen_FU2["ID"]=Imagen_FU2.ID_long.str.split(pat="_", expand=True)[0]
Imagen_FU3["ID"]=Imagen_FU3.ID_long.str.split(pat="_", expand=True)[0]

Imagen_BL["ID_visit"] = 1
Imagen_FU2["ID_visit"] = 2
Imagen_FU3["ID_visit"] = 3
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

#%%

df_IMAGEN_age_sex_site['ID'] = df_IMAGEN_age_sex_site['ID'].astype(str).str.zfill(12)

df_IMAGEN_BL_full = df_IMAGEN_age_sex_site.merge(Imagen_BL, on = "ID")
#%% IMAGEN 2

# Age and sex
df_IMAGEN_FU2 = pd.read_csv("/project_cephfs/3022017.06/IMAGEN/phenotypes/FU2/meta_data/IMAGEN_age_at_mri_acquisition_FU2.csv")
df_IMAGEN_FU2 = df_IMAGEN_FU2.rename(columns={"DTI":"age", "PSC2":"ID"})
df_IMAGEN_FU2_age = df_IMAGEN_FU2[["age", "ID"]]
df_IMAGEN_FU2_age["age"]=df_IMAGEN_FU2_age["age"]/365

#%%
df_IMAGEN_FU2_age_sex = df_IMAGEN_FU2_age.merge(df_IMAGEN_sex, on ="ID", how = "outer")
df_IMAGEN_FU2_age_sex["ID"] = df_IMAGEN_age_sex['ID'].astype(str).str.zfill(12)

df_IMAGEN_FU2_full = df_IMAGEN_FU2_age_sex.merge(Imagen_FU2, on = "ID", how = "right")

#%% IMAGEN 3
df_IMAGEN_FU3 = pd.read_csv("/project_cephfs/3022017.06/IMAGEN/phenotypes/FU3/dawba/IMAGEN_dawba_FU3.tsv", sep="\t", index_col=False)
df_IMAGEN_FU3 =  df_IMAGEN_FU3.rename(columns={"PSC2":"ID", "gender":"sex"})
df_IMAGEN_FU3 = df_IMAGEN_FU3[["ID", "age", "sex"]]
df_IMAGEN_FU3['ID'] = df_IMAGEN_FU3['ID'].astype(str).str.zfill(12)


df_IMAGEN_FU3_full = df_IMAGEN_FU3.merge(Imagen_FU3, on ="ID", how ="outer")

#%%
#%% Euler number
euler_bl = pd.read_csv('/project_cephfs/3022017.06/IMAGEN/subjects_v2.7/BL/imaging/freesurfer/euler.tsv', sep = "\t")
euler_FU2 = pd.read_csv('/project_cephfs/3022017.06/IMAGEN/subjects_v2.7/FU2/imaging/freesurfer/euler.tsv', sep = "\t")
euler_FU3 = pd.read_csv('/project_cephfs/3022017.06/IMAGEN/subjects_v2.7/FU3/imaging/freesurfer/euler.tsv', sep = "\t")

euler_bl= euler_bl.rename(columns={"orig.nofix": "ID", "rheno":"rh_en", "lheno": "lh_en", "mean":"avg_en"})
euler_FU2 = euler_FU2.rename(columns={"orig.nofix": "ID", "rheno":"rh_en", "lheno": "lh_en", "mean":"avg_en"})
euler_FU3 = euler_FU3.rename(columns={"orig.nofix": "ID", "rheno":"rh_en", "lheno": "lh_en", "mean":"avg_en"})

#%%
#Imagen_bl_all=Imagen_bl_full.merge(euler_bl, on="ID", how="outer")
#df_IMAGEN_FU2_all = df_IMAGEN_FU2_full.merge(euler_FU2, on="ID", how="outer")
#df_IMAGEN_FU3_all = df_IMAGEN_FU3_full.merge(euler_FU3, on="ID", how="outer")

#%%
df_IMAGEN_site= df_IMAGEN_age_sex_site[['ID', "site", "site_id"]]

df_IMAGEN_FU2_full = df_IMAGEN_FU2_full.merge(df_IMAGEN_site, on = "ID")
df_IMAGEN_FU3_full = df_IMAGEN_FU3_full.merge(df_IMAGEN_site, on = "ID")

#%%
IMAGEN=pd.concat([df_IMAGEN_BL_full, df_IMAGEN_FU2_full, df_IMAGEN_FU3_full])

IMAGEN["sex"] = IMAGEN["sex"].replace(2, 0)
#%%
IMAGEN["sex"] = pd.to_numeric(IMAGEN["sex"], errors="coerce")

#%%
IMAGEN['ID_long'] = IMAGEN['ID_long'].astype(str)
IMAGEN['processing_type'] = IMAGEN['ID_long'].apply(
    lambda x: "long" if '.long.' in x else "cross"
)

IMAGEN["ID_subject"] = IMAGEN["ID"]

IMAGEN_cross = IMAGEN[IMAGEN["processing_type"]=="cross"]
IMAGEN_long = IMAGEN[IMAGEN["processing_type"]=="long"]

assert_few_nans(IMAGEN_cross)
assert_few_nans(IMAGEN_long)

assert_sex_column_is_binary(IMAGEN_cross)
assert_sex_column_is_binary(IMAGEN_long)

assert_no_negatives(IMAGEN_cross, idp_cols)
assert_no_negatives(IMAGEN_long, idp_cols)



if write:
    IMAGEN.to_csv(os.path.join(data_out_dir,"csv/IMAGEN_cross_SC.csv"),index=False )
    IMAGEN.to_pickle(os.path.join(data_out_dir, "IMAGEN_cross_SC.pkl"))
    IMAGEN.to_csv(os.path.join(data_out_dir,"csv/IMAGEN_long_SC.csv"),index=False )
    IMAGEN.to_pickle(os.path.join(data_out_dir, "IMAGEN_long_SC.pkl"))
#%% ABCD 
# Male = 1, Female =2
# Race  1 = White; 2 = Black; 3 = Hispanic; 4 = Asian; 5 = Other

# needs recoding

ABCD_race = pd.read_csv('/project_cephfs/3022017.02/phenotypes/ABCD_AnnualRelease5.0/core/abcd-general/abcd_p_demo.csv'
                        , usecols =["race_ethnicity", "eventname", "src_subject_id"])
ABCD = pd.read_csv('/project_cephfs/3022017.02/phenotypes/ABCD_AnnualRelease5.0/core/abcd-general/abcd_y_lt.csv')

ABCD_stats = pd.read_csv('/project_cephfs/3022017.02/freesurfer/aseg_stats_all.txt', sep ="\t")
ABCD_stats = ABCD_stats.rename(columns={"Measure:volume":"ID_long"})

#%%
#ABCD_euler = pd.read_csv('/project_cephfs/3022017.02/freesurfer/euler_number.csv')

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

ABCD_demo = ABCD_demo.rename(columns={"interview_age":"age", "src_subject_id":"ID_subject", "site_id_l":"site"})

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

ref = ABCD_sex.groupby('src_subject_id')['demo_sex_v2'].first()
ABCD_sex['demo_sex_v2'] = ABCD_sex['demo_sex_v2'].fillna(ABCD_sex['src_subject_id'].map(ref))

ABCD_sex = ABCD_sex.rename(columns={"demo_sex_v2":"sex", "src_subject_id":"ID_subject"})

#%%
ABCD_stats["ID"]=ABCD_stats["ID_long"].str.split(pat ="_", expand=True)[0]

#%%
# Split the 'ID_long' column
split_columns = ABCD_stats["ID_long"].str.split(pat="-", expand=True)

# Filter rows where the third part [2] is not None or empty
ABCD_stats = ABCD_stats[split_columns[2].notna() & (split_columns[2] != '')]
ABCD_stats["eventname"] = ABCD_stats["ID_long"].str.split(pat = "-", expand=True)[2]


ABCD_stats["ID_visit"] = ABCD_stats["eventname"]

ABCD_stats['processing_type'] = ABCD_stats['eventname'].apply(
    lambda x: "long" if '.long.' in x else "cross"
)

ABCD_stats = ABCD_stats.replace({'ID_visit':{'4YearFollowUpYArm1*':'4', 'baselineYear1Arm1*':'1', 
                                   '2YearFollowUpYArm1*':'2', '3YearFollowUpYArm1*':'3'}}, regex = True)

ABCD_stats['ID_visit'] = ABCD_stats['ID_visit'].str.replace('.long.sub', '', regex=False)
ABCD_stats = ABCD_stats.rename(columns={"ID":"ID_subject",})

#%% Divide into long and cross
# cross
ABCD_stats_cross = ABCD_stats[ABCD_stats["processing_type"] =="cross"]
ABCD_stats_long = ABCD_stats[ABCD_stats["processing_type"] =="long"]

ABCD_demo = ABCD_demo[["site", "age", "ID"]]
ABCD_full_cross = ABCD_stats_cross.merge(ABCD_demo, left_on = "ID_long", right_on = "ID")

ABCD_sex = ABCD_sex[["sex", "ID"]]
ABCD_full_cross = ABCD_full_cross.merge(ABCD_sex, left_on = "ID", right_on = "ID")
#long
ABCD_stats_long["ID"] = ABCD_stats_long["ID_long"].str.split(pat = ".", expand=True)[0]
ABCD_full_long = ABCD_stats_long.merge(ABCD_demo, left_on = "ID", right_on = "ID")
ABCD_full_long = ABCD_full_long.merge(ABCD_sex, left_on = "ID", right_on = "ID")
# recode sex

ABCD_full_cross.sex.replace(2, 0, inplace=True)
ABCD_full_long.sex.replace(2, 0, inplace=True)


ABCD_full_cross["site_id"] = "540" + ABCD_full_cross["site"].str.replace("site", "", regex=True)
ABCD_full_cross["site"] = "abcd_" + ABCD_full_cross["site"]

ABCD_full_long["site_id"] = "540" + ABCD_full_long["site"].str.replace("site", "", regex=True)
ABCD_full_long["site"] = "abcd_" + ABCD_full_long["site"]

ABCD_full_long["sex"] = pd.to_numeric(ABCD_full_long["sex"], errors="coerce")
ABCD_full_cross["sex"] = pd.to_numeric(ABCD_full_cross["sex"], errors="coerce")

ABCD_full_long = ABCD_full_long[ABCD_full_long['sex'].isin([0, 1])]
ABCD_full_cross = ABCD_full_cross[ABCD_full_cross['sex'].isin([0, 1])]

assert_few_nans(ABCD_full_cross)
assert_few_nans(ABCD_full_long)

assert_sex_column_is_binary(ABCD_full_cross)
assert_sex_column_is_binary(ABCD_full_long)

assert_no_negatives(ABCD_full_cross, idp_cols)
assert_no_negatives(ABCD_full_long, idp_cols)

if write:
    ABCD_full_long.to_pickle(os.path.join(data_out_dir,"ABCD_long_SC.pkl"))
    ABCD_full_cross.to_pickle(os.path.join(data_out_dir,"ABCD_cross_SC.pkl"))
    #ABCD.to_pickle(os.path.join(data_out_dir,"ABCD.pkl"))
#%% UKB
# Sex encoding: Female = 0, Male =1
# Race encoding: {1001 : 1, 1002 : 1, 1003 : 1, 2000 : 2, 
#                           2001 : 2, 2003 : 2, 2004 : 2, 2002 : 2, 
#                           3000 : 3, 3001 : 3, 3002 : 3, 3003 : 3, 3004 : 3,
#                           4000 : 4, 4001 : 4, 4002 : 4, 4003 : 4,
#                           5000 : 5, 6000 : 6}}

# 1: White
# 2: Mixed
# 3: Asian
# 4: Black

# is recoded to:
    
# 1: White
# 2: Asian
# 3: Black
# > 4: Other?Mixed
    
      
UKB_race = pd.read_csv("/project_cephfs/3022017.05/phenotypes/current/02_genetics.csv", usecols = ["eid","21000-0.0"])
UKB_age1 = pd.read_csv("/project_cephfs/3022017.05/phenotypes/age_sex_site_ukb35187.csv")
UKB_age2 = pd.read_csv("/project_cephfs/3022017.05/phenotypes/age_sex_site_ukb42006.csv")

UKB_age1.rename(columns={"eid":"ID","31-0.0":"sex", "54-2.0":"site", "21003-2.0":"age"}, inplace = True)

UKB_age2.rename(columns={"eid":"ID_subject","31-0.0":"sex", "54-2.0":"site","54-3.0":"site2", "21003-2.0":"age1","21003-3.0":"age2"}, inplace = True)
UKB_age2["ID_subject"] = UKB_age2["ID_subject"].astype(str)


UKB = pd.read_csv(os.path.join(base_dir,"Velocity/DATA_pre_processed","aseg_stats.txt"), sep = "\t")
UKB["Left-Thalamus"] = UKB["Left-Thalamus"] +UKB["Left-Thalamus-Proper"]
UKB["Right-Thalamus"] = UKB["Right-Thalamus"] +UKB["Right-Thalamus-Proper"]

UKB = UKB.rename(columns={"Measure:volume":"ID"})

#%% convert race
assert UKB_race["21000-0.0"].dtype == 'float64'

UKB_race["21000-0.0"].unique()
UKB_race["race_ethnicity"] = UKB_race["21000-0.0"]
UKB_race.replace({'race_ethnicity': {1001 : 1, 1002 : 1, 1003 : 1, 2000 : 4, 
                           2001 : 4, 2003 : 4, 2004 : 4, 2002 : 4, 
                           3000 : 2, 3001 : 2, 3002 : 2, 3003 : 2, 3004 : 2,
                           4000 : 3, 4001 : 3, 4002 : 3, 4003 : 3,
                           5000 : 5, 6000 : 6}}, inplace =True)
UKB_race["race_ethnicity"].unique()
UKB_race.rename(columns={"eid":"ID_subject"}, inplace = True)
UKB_race.drop(columns={"21000-0.0"}, inplace=True)


#%%

UKB['processing_type'] = UKB['ID'].apply(
    lambda x: "long" if 'long' in x else "cross"
)

#%% subset

UKB_cross = UKB[UKB["processing_type"]=="cross"]
UKB_long = UKB[UKB["processing_type"]=="long"]

# remove tenplate
UKB_long = UKB_long[UKB_long['ID'].str.contains(r'\.long\.', na=False)]

UKB_long['ID_visit'] = UKB_long['ID'].apply(
    lambda x: "2" if 'scan2' in x else "1"
)

UKB_cross['ID_visit'] = UKB_cross['ID'].apply(
    lambda x: "2" if 'scan2' in x else "1"
)

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

#cross
UKB_cross["ID_subject"] = UKB_cross["ID"].str.split(pat="_", expand = True)[0]
UKB_cross[["ID_subject", "ID_visit"]] =  UKB_cross[["ID_subject", "ID_visit"]].astype(int)
UKB_age[["ID_subject", "ID_visit"]] =  UKB_age[["ID_subject", "ID_visit"]].astype(int)

#long
UKB_long["ID_subject"] = UKB_long["ID"].str.split(pat=".", expand = True)[2]
UKB_long["ID_subject"]=UKB_long["ID_subject"].str.replace("_long", "", regex=True)
UKB_long[["ID_subject", "ID_visit"]] =  UKB_long[["ID_subject", "ID_visit"]].astype(int)
#%%
UKB_cross = UKB_cross.merge(UKB_age, on = ["ID_subject", "ID_visit"])
UKB_long = UKB_long.merge(UKB_age, on = ["ID_subject", "ID_visit"])

UKB_cross["site_id"] = UKB_cross["site"]
UKB_long["site_id"] = UKB_long["site"]
#%% merge race in;
UKB_cross["ID_subject"] =  UKB_cross["ID_subject"].astype(int)
UKB_long["ID_subject"] =  UKB_long["ID_subject"].astype(int)

assert_few_nans(UKB_cross)
assert_few_nans(UKB_long)

assert_no_negatives(UKB_cross, idp_cols)
assert_no_negatives(UKB_long, idp_cols)

assert_sex_column_is_binary(UKB_cross)
assert_sex_column_is_binary(UKB_long)


if write:
    UKB_cross.to_pickle(os.path.join(data_out_dir, "UKB_cross_SC.pkl"))
    UKB_long.to_pickle(os.path.join(data_out_dir, "UKB_long_SC.pkl"))

#%% HCP Babies
# Sex encoding: 0: Female, 1: Male
# Race encoding: White', 'Asian', 'More than one race','Black or African American'

# recoding to White: 1, Asian: 2, Black or African American : 3, other/more: 4

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

HCP_Babies = pd.read_csv("/project_cephfs/3022017.06/HCP_Baby/freesurfer/aseg_stats.txt", sep = "\t")
HCP_Babies = HCP_Babies.rename(columns={"Measure:volume":"ID"})

HCP_babies_ids = pd.read_csv('/project_cephfs/3022017.06/HCP_Baby/phenotypes/ndar_subject01.txt', sep = "\t",
                             usecols= ["src_subject_id", "race", "ethnic_group"])

HCP_babies_ids = HCP_babies_ids.iloc[1:]

#HCP_babies_ids.race.unique()


#%% 

HCP_Babies['processing_type'] = HCP_Babies['ID'].apply(
    lambda x: "long" if 'long' in x else "cross"
)

HCP_Babies['ID_subject'] = HCP_Babies['ID'].apply(
    lambda x: x.split('.long.')[0] if '.long.' in x else x 
)

HCP_Babies = HCP_Babies[~HCP_Babies['ID_subject'].str.endswith('_long', na=False)] 
#%%
HCP_Babies = HCP_Babies.merge(HCP_babies_pass, right_on = "ID", left_on = "ID_subject")

HCP_Babies.rename(columns={"ID_x":"ID_long"}, inplace = True)
HCP_Babies=HCP_Babies.drop(columns="ID_y")
#%%
HCP_Babies["src_subject_id"] = HCP_Babies.ID_long.str.split(pat = "_", expand=True)[0]
HCP_Babies["src_subject_id"] = HCP_Babies["src_subject_id"].replace("MNBCP", "", regex=True)
HCP_Babies["src_subject_id"] = HCP_Babies["src_subject_id"].replace("NCBCP", "", regex=True)
HCP_Babies["src_subject_id"] = HCP_Babies["src_subject_id"].replace("MNBC", "", regex=True)


test2=HCP_Babies["ID_long"].str.split(pat="-", expand=True)[3]
HCP_Babies["scan_date"]=test2.str.split(pat=".", expand=True)[0]
HCP_Babies["ID_visit"] = HCP_Babies["src_subject_id"] + "_"+ HCP_Babies["scan_date"]


#%%
HCP_Babies_age = HCP_Babies_age[["ID_visit", "age"]]
HCP_Babies_age.drop_duplicates(inplace=True)
HCP_Babies = HCP_Babies.merge(HCP_Babies_age,  on="ID_visit", how ="left")
HCP_babies_ids = HCP_babies_ids.drop_duplicates()
#%%
#HCP_Babies= HCP_Babies.merge(HCP_babies_ids, on="src_subject_id", how = "left")

#%%
#HCP_Babies["site"]= "U"+HCP_Babies["ID"].str[:2]

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

#HCP_Babies.drop(columns=["race", "ethnic_group"], inplace = True)

#%%
HCP_Babies = HCP_Babies.dropna(how = 'any')
#HCP_Babies = HCP_Babies[~HCP_Babies["ID"].str.contains('long')]
HCP_Babies["site_id"] = 9001

HCP_Babies["ID_visit"] = HCP_Babies["ID_long"].str.split(pat="-", expand=True)[0]
HCP_Babies["ID_visit"] = HCP_Babies["ID_visit"].str.split(pat="_", expand=True)[1]
HCP_Babies["ID_visit"] = HCP_Babies["ID_visit"].str.replace("v0", "")
HCP_Babies["ID_visit"] = HCP_Babies["ID_visit"].astype(int)
#%%
HCP_Babies = HCP_Babies.rename(columns={"ID_subject":"ID_subject_long"})
HCP_Babies = HCP_Babies.rename(columns={"src_subject_id":"ID_subject"})

#%% race recoding

#HCP_Babies.race= HCP_Babies.race.str.replace('White', '1')
#HCP_Babies.race =  HCP_Babies.race.str.replace('Asian', '2') # there were no African Americans
#HCP_Babies.race =  HCP_Babies.race.str.replace('More than one race', '4')

#HCP_Babies.rename(columns={"race":"race_ethnicity"}, inplace = True)

#HCP_Babies.rename(columns={"age_x":"age"}, inplace = True)
#HCP_Babies=HCP_Babies.drop(columns="age_y")

#%%
HPC_Babies_cross = HCP_Babies[HCP_Babies["processing_type"] =="cross"]
HPC_Babies_long = HCP_Babies[HCP_Babies["processing_type"] =="long"]



if write:
    HPC_Babies_cross.to_pickle(os.path.join(data_out_dir, "HCP_Babies_cross_sc.pkl"))
    HPC_Babies_long.to_pickle(os.path.join(data_out_dir, "HCP_Babies_long_sc.pkl"))
#%% NIH Babies
# are already in the data set
#%%
# age, sex, race, visit_id, subject_id. site_id. site, site_index
# 'White', 'White/Not Provided', nan,
#        'American Indian or Alaskan Native/White',
#        'African American or Black',
#        'African American or Black/American Indian or Alaskan Native/White',
#        'Asian', 'Not Provided', 'African American or Black/White',
#        'Asian/Not Provided',
#        'African American or Black/American Indian or Alaskan Native',
#        'Asian/White', 'Native Hawaiian or Other Pacific Islander/White',
#        'African American or Black/Asian',
#        'African American or Black/Not Provided',
#        'American Indian or Alaskan Native'

# race" white: 1, Asian: 2, African American or Black : 3, 


NIH_Babies  = pd.read_csv("/project_cephfs/3022017.06/NIH_pedMRI/freesurfer/aseg_stats.txt", sep = "\t")
NIH_Babies.rename(columns={"Measure:volume":"ID"}, inplace=True)


NIH_Babies_cross = NIH_Babies[~NIH_Babies["ID"].str.contains("long")]
NIH_Babies_long = NIH_Babies[NIH_Babies["ID"].str.contains("long")]


#%%
NIH_Babies_cross["ID_subject"] = NIH_Babies_cross["ID"].str.split(pat="_", expand=True)[0]
NIH_Babies_cross["ID_visit"] = NIH_Babies_cross["ID"].str.split(pat="_", expand=True)[1]

#%%
NIH_Babies_long["ID_subject"] = NIH_Babies_long["ID"].str.split(pat="_", expand=True)[0]
NIH_Babies_long["ID_visit"] = NIH_Babies_long["ID"].str.split(pat="_", expand=True)[1]
#%%
NIH_Babies_cross["ID_visit"] = NIH_Babies_cross["ID_visit"].str.replace("v", "")
NIH_Babies_long["ID_visit"] = NIH_Babies_long["ID_visit"].str.split(pat=".", expand=True)[0]
NIH_Babies_long["ID_visit"] = NIH_Babies_long["ID_visit"].str.replace("v", "")

#%%
NIH_Babies_long = NIH_Babies_long[~NIH_Babies_long["ID_visit"].str.contains("long")]

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
#NIH_Babies["ID_visit"] = NIH_Babies["ID_visit"].astype(str)

NIH_Babies_demo["ID_subject"] = NIH_Babies_demo["ID_subject"].astype(str)
NIH_Babies_demo["ID"] = NIH_Babies_demo["ID_subject"] + "_" + NIH_Babies_demo["ID_visit"]
NIH_Babies_demo = NIH_Babies_demo[~NIH_Babies_demo["ID_visit"].str.contains("I")]
NIH_Babies_demo = NIH_Babies_demo[["ID", "age", "site_id", "site", "sex"]]

#%%
NIH_Babies_cross["ID_visit"] = NIH_Babies_cross["ID_visit"].astype(str)
NIH_Babies_long["ID_visit"] = NIH_Babies_long["ID_visit"].astype(str)

NIH_Babies_cross = NIH_Babies_cross.reset_index(drop=True)
NIH_Babies_long = NIH_Babies_long.reset_index(drop=True)

NIH_Babies_demo = NIH_Babies_demo.reset_index(drop=True)

NIH_Babies_long.rename(columns={"ID":"ID_long"}, inplace=True)

NIH_Babies_cross["ID"]=NIH_Babies_cross["ID_subject"] + "_" + NIH_Babies_cross["ID_visit"]
NIH_Babies_long["ID"]=NIH_Babies_long["ID_subject"] + "_" + NIH_Babies_long["ID_visit"]

#%% merge
NIH_Babies_cross = NIH_Babies_cross.merge(NIH_Babies_demo, on = "ID")
NIH_Babies_long = NIH_Babies_long.merge(NIH_Babies_demo, on = "ID")
#%%
# #fill nans in race

# race_ref = NIH_Babies.groupby('ID_subject')['race'].first()
# NIH_Babies['race'] = NIH_Babies['race'].fillna(NIH_Babies['ID_subject'].map(race_ref))

# #%% race backcoding
# NIH_Babies["race"]=NIH_Babies["race"].astype(str)

# NIH_Babies["race"] = NIH_Babies["race"].str.replace('White', '1')
# NIH_Babies["race"] = NIH_Babies["race"].str.replace('Asian', '2')
# NIH_Babies["race"] = NIH_Babies["race"].str.replace('African American or Black', '3')

# filter = NIH_Babies['race'].str.len() > 1

# NIH_Babies.loc[filter,'race'] = '4'
# NIH_Babies = NIH_Babies.rename(columns={"race":"race_ethnicity"})



                        
if write:
    NIH_Babies_cross.to_pickle(os.path.join(data_out_dir, "NIH_Babies_cross_sc.pkl"))
    NIH_Babies_long.to_pickle(os.path.join(data_out_dir, "NIH_Babies_long_sc.pkl"))


  