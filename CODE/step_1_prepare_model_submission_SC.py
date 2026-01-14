#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:44:09 2024

@author: johbay

This file prepares the 
"""

#%% Create cross matched
# load 
from utils import ldpkl
import numpy as np
import os
from utils import SC_idp_cols, reduced_sc_idp_cols, remove_outliers_and_adjust, count_site_occurrences
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from utilities_test import *
import random
#%%

data_in_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC'
base_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/'
#%%
write = True
#%%

UKB_full_cross = pd.read_pickle(os.path.join(data_in_dir, "UKB_cross_SC.pkl"))
IMAGEN_full_cross = pd.read_pickle(os.path.join(data_in_dir, "IMAGEN_cross_SC.pkl")) # contains both
ABCD_full_cross = pd.read_pickle(os.path.join(data_in_dir, "ABCD_cross_SC.pkl"))
OASIS2_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_cross_SC.pkl"))
OASIS3_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_cross_SC.pkl"))
#HCP_Babies_full_cross = pd.read_pickle(os.path.join(data_in_dir, "HCP_Babies_cross_SC.pkl"))
#NIH_Babies_full_cross = pd.read_pickle(os.path.join(data_in_dir, "NIH_Babies_cross_SC.pkl"))
ADNI_adaptation_train = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/ADNI_SC_adaptation_train.pkl')


UKB_full_long = pd.read_pickle(os.path.join(data_in_dir, "UKB_long_SC.pkl"))
IMAGEN_full_long = pd.read_pickle(os.path.join(data_in_dir, "IMAGEN_long_SC.pkl")) #
ABCD_full_long = pd.read_pickle(os.path.join(data_in_dir, "ABCD_long_SC.pkl"))
OASIS3_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_SC.pkl"))
OASIS2_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_SC.pkl"))
#HCP_Babies_full_long = pd.read_pickle(os.path.join(data_in_dir, "HCP_Babies_long_SC.pkl"))
#NIH_Babies_full_long = pd.read_pickle(os.path.join(data_in_dir, "NIH_Babies_long_SC.pkl"))
ADNI_adapttaion_test = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/ADNI_SC_adaptation_test.pkl')
ADNI_clinical = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/ADNI_SC_clinical_goodsites.pkl')



life_tr= pd.read_csv(os.path.join(base_dir, "Lifedata_with_PING/lifespan_big_controls_sc_extended_tr.csv"), dtype={0: str, 4: str})
life_te = pd.read_csv(os.path.join(base_dir,"Lifedata_with_PING/lifespan_big_controls_sc_extended_te.csv" ), dtype={0: str, 4: str})
life_tr["site"].unique()
life_te["site"].unique()

#%% load demented files
OASIS2_full_cross_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_cross_SC_demented.pkl"))
OASIS2_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_SC_demented.pkl"))

OASIS3_full_cross_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_cross_SC_demented.pkl"))
OASIS3_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_SC_demented.pkl"))




#%% look at difference between demented and non-demented 
for _, group_data in OASIS2_full_long.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color='black', alpha=0.3)
for group_name, group_data in OASIS2_full_long_demented.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color="red", alpha = 0.7)



for _, group_data in OASIS3_full_long.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color='black', alpha=0.3)
for group_name, group_data in OASIS3_full_long_demented.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color="red", alpha = 0.7)

plt.show()

#%%
#%% look at difference between demented and non-demented 
for _, group_data in OASIS2_full_cross.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color='black', alpha=0.3)
for group_name, group_data in OASIS2_full_cross_demented.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color="red", alpha = 0.7)



for _, group_data in OASIS3_full_cross.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color='black', alpha=0.3)
for group_name, group_data in OASIS3_full_cross_demented.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color="red", alpha = 0.7)

plt.show()

#%%
idp_cols = SC_idp_cols()
additional_cols = ["ID_subject","age", "sex", "site_id","ID_visit"]

cols = idp_cols +additional_cols
#cols.remove('Left-Thalamus')
#cols.remove('Right-Thalamus')

#%%
IMAGEN_long = IMAGEN_full_long[cols]
UKB_long = UKB_full_long[cols]
ABCD_long = ABCD_full_long[cols]
OASIS2_long = OASIS2_full_long[cols]
OASIS3_long = OASIS3_full_long[cols]
ADNI_clinical = ADNI_clinical[cols]
ADNI_adaptation_test = ADNI_adapttaion_test[cols]
#HCP_Babies_long = HCP_Babies_full_long[cols] 
#NIH_Babies_long = NIH_Babies_full_long[cols] 

IMAGEN_cross = IMAGEN_full_cross[cols]
UKB_cross = UKB_full_cross[cols]
ABCD_cross = ABCD_full_cross[cols]
OASIS2_cross = OASIS2_full_cross[cols]
OASIS3_cross = OASIS3_full_cross[cols]
ADNI_adaptation_train = ADNI_adaptation_train[cols]
#HCP_Babies_cross = HCP_Babies_full_cross[cols] 
#NIH_Babies_cross = NIH_Babies_full_cross[cols] 

#%%
OASIS3_cross_demented = OASIS3_full_cross_demented[cols]
OASIS3_long_demented = OASIS3_full_long_demented[cols]

OASIS2_cross_demented = OASIS2_full_cross_demented[cols]
OASIS2_long_demented = OASIS2_full_long_demented[cols]

OASIS23_long_demented = pd.concat([OASIS2_long_demented, OASIS3_long_demented])
 #%% write out the demented files for predict - not necessary in new version

# X_test_OASIS23_long_demented = OASIS23_long_demented[["age","sex"]].to_numpy(dtype=float)
# Z_test_OASIS23_long_demented = OASIS23_long_demented["site_id"].to_numpy(dtype=float)
# #Y_test= test[measure_name].to_numpy()
# Y_test_OASIS23_long_demented= OASIS23_long_demented[idp_cols].to_numpy(dtype=float)


# os.chdir(base_dir)

# if write:
#     with open('Velocity/DATA_NM/X_test_OASIS23_long_demented2.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(X_test_OASIS23_long_demented), file)
#     with open('Velocity/DATA_NM/Y_test_OASIS23_long_demented2.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_test_OASIS23_long_demented), file)
#     with open('Velocity/DATA_NM/tebefile_OASIS23_long_demented2.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Z_test_OASIS23_long_demented), file)



#%%
life_tr["ID_visit"] = 1
life_te["ID_visit"] = 1

life_tr.rename(columns={'Unnamed: 0':'ID', 'Left-Thalamus-Proper':'Left-Thalamus', 'Right-Thalamus-Proper':'Right-Thalamus'}, inplace=True)
life_te.rename(columns={'Unnamed: 0':'ID', 'Left-Thalamus-Proper':'Left-Thalamus', 'Right-Thalamus-Proper':'Right-Thalamus'}, inplace=True)

life_tr["ID_subject"] = life_tr["ID"]
life_te["ID_subject"] = life_te["ID"]


#%%
#data_cross = [IMAGEN_cross, OASIS2_cross, UKB_cross, ABCD_cross, OASIS3_cross, HCP_Babies_cross, NIH_Babies_cross]
#data_cross = [IMAGEN_cross, OASIS2_cross, UKB_cross, ABCD_cross, OASIS3_cross]
data_cross = [IMAGEN_cross, OASIS2_cross, UKB_cross, ABCD_cross, OASIS3_cross]
data_cross = pd.concat(data_cross)

data_cross = data_cross.dropna(how='any')

#data_long = [IMAGEN_long, OASIS2_long, UKB_long, ABCD_long, OASIS3_long, HCP_Babies_long, NIH_Babies_long]
data_long = [IMAGEN_long, OASIS2_long, UKB_long, ABCD_long, OASIS3_long]
data_long = pd.concat(data_long)

data_long = data_long.dropna(how='any')

assert_few_nans(data_long)
assert_few_nans(data_cross)
assert_no_negatives(data_long, idp_cols)
assert_no_negatives(data_cross, idp_cols)
assert_sex_column_is_binary(data_long)
assert_sex_column_is_binary(data_cross)


if write:
    data_long.to_pickle(os.path.join(data_in_dir,"data_long_SC_adults.pkl"))
    data_long.to_csv(os.path.join(data_in_dir, "data_long_SC_adults.csv"),index=False )
    data_cross.to_pickle(os.path.join(data_in_dir,"data_cross_SC_adults.pkl"))
    data_cross.to_csv(os.path.join(data_in_dir, "data_cross_SC_adults.csv"),index=False )


#%% Cross sectional data
drop_sites = ['ukb-11027.0', 'ukb-11025.0', 'Oasis2',
       'Oasis3', 'ABCD_06', 'ABCD_10', 'ABCD_19', 'ABCD_01',
       'ABCD_02', 'ABCD_17', 'ABCD_14', 'ABCD_13', 'ABCD_11', 'ABCD_07',
       'ABCD_08', 'ABCD_09', 'ABCD_15', 'ABCD_20', 'ABCD_16', 'ABCD_04',
       'ABCD_21', 'ABCD_03', 'ABCD_18', 'ABCD_12', 'ABCD_05','HCP_Baby_UMN', 'HCP_Baby_UNC',
       'NIH_pedMRI_4', 'NIH_pedMRI_3', 'NIH_pedMRI_1', 'NIH_pedMRI_2',
       'NIH_pedMRI_5', 'NIH_pedMRI_6']

#%%
life_tr_bare=life_tr[~life_tr["site"].str.contains("|".join(drop_sites))]
life_te_bare=life_te[~life_te["site"].str.contains("|".join(drop_sites))]
#%%
life_tr_bare["site"].unique()
life_tr_bare.loc[:,'site_id'] = pd.factorize(life_tr_bare['site'])[0]
life_tr_bare.loc[:,'site_id'] = life_tr_bare['site_id'] +1
life_tr_bare.loc[:,'site_id'] = life_tr_bare['site_id'] +500

ref = life_tr_bare.groupby('site')['site_id'].first()
life_te_bare.loc[:, 'site_id'] = life_te_bare['site'].map(ref)


test = life_tr_bare["site"].value_counts()
test2 = life_te_bare["site"].value_counts()
#%%
life_te_bare = life_te_bare[cols]
life_tr_bare = life_tr_bare[cols]

life_te_bare = life_te_bare[~life_te_bare["site_id"].isna()]
life_tr_bare = life_tr_bare[~life_tr_bare["site_id"].isna()]

#%%
life_tr_bare['site_id'].value_counts()
life_te_bare['site_id'].value_counts()

#%%
life_tr_bare["which_dataset"] = "1"
life_te_bare["which_dataset"] = "1"


#%%
# split the longitudinal data set
data_long.groupby(by="site_id").count()


data_cross['ID_visit'] = data_cross["ID_visit"].astype(float)
data_cross['ID_visit'] = data_cross["ID_visit"].astype(str)
data_cross["ID_visit"].unique()

data_long['ID_visit'] = data_long["ID_visit"].astype(float)
data_long['ID_visit'] = data_long["ID_visit"].astype(str)
data_long["ID_visit"].unique()
#%%
data_long["ID_subject"] = data_long["ID_subject"].astype(str)

#%%
subs_long=data_long["ID_subject"].unique()

#%%
data_long["which_dataset"] = "2"
#test_long["which_dataset"] = "2"

data_long.shape[0]
subs=data_long["ID_subject"].unique()
#%% remove longitudinal subjects from the cross sectional data set
data_cross["ID_subject"] = data_cross["ID_subject"].astype(str)

bool_cross = data_cross["ID_subject"].str.contains("|".join(subs_long))
#te_bool_cross = data_cross["ID_subject"].str.contains("|".join(subs_te))
bool_non_cross = ~bool_cross

data_cross_non_long = data_cross[bool_non_cross] 

#%%
data_cross_non_long.loc[:,"ID_subject"] = data_cross_non_long["ID_subject"].astype(str)
data_cross_non_long = data_cross_non_long.dropna(how='any')

cross_subs = data_cross_non_long["ID_subject"]
#cross_subs["ID_subject"] = cross_subs["ID_subject"].astype(str)

rng = np.random.default_rng(seed=42)
tr_cross = rng.random(cross_subs.shape[0]) > 0.4
te_cross = ~tr_cross

tr_data_cross_non_long = data_cross_non_long.copy()[tr_cross]
te_data_cross_non_long = data_cross_non_long.copy()[te_cross]

#%%
tr_data_cross_non_long.loc[:, "which_dataset"] = "3"
te_data_cross_non_long.loc[:,"which_dataset"] = "3"
#%%
train = pd.concat([life_tr_bare, tr_data_cross_non_long])
test = pd.concat([life_te_bare, te_data_cross_non_long, data_long])

#%%
test3 =set(data_long["ID_subject"]) & set(tr_data_cross_non_long["ID_subject"])

#make sure that all the sites in the training set are also in the test set
train_sites = train['site_id'].value_counts()
test_sites = test['site_id'].value_counts()

train_sites = train["site_id"].unique()
test_sites = test["site_id"].unique()

train = train.drop_duplicates()
test = test.drop_duplicates()

difference = [obj for obj in train_sites if obj not in test_sites]
print(difference)
difference = [obj for obj in test_sites if obj not in train_sites]
print(difference)

#%% these sites are all only in the test set and some need to be added to the 
# # training set
# site_800 = test[test["site_id"].astype(str).str.contains("800", na=False)]

# id_list = site_800["ID_subject"].unique()
# tr = rng.random(id_list.shape[0]) < 0.1
# transfer = id_list[tr]

# transfer_subjects = test[test["ID_subject"].isin(transfer)]

# test = test[~test["ID_subject"].isin(transfer)]
# train = pd.concat([train, transfer_subjects])
# train2 = train

#%%
test_only_sites = [404.0, 700, 702, 701, 703, 707, 706, 704, 705, 800]

# 1. Filter test subjects from those sites
test_subset = test[test["site_id"].isin(test_only_sites)]

# 2. Sample 30% of them
to_move = test_subset.sample(frac=0.1, random_state=42)

# 3. Get subject IDs being moved (for tracking)
moved_subject_ids = to_move["ID_subject"].tolist()
print("✅ Moving these subject IDs to training set:")
print(moved_subject_ids)

# 4. Move them
test = test.drop(to_move.index)
train= pd.concat([train, to_move], ignore_index=True)

#%% write the models that will be fully trained
train_retrain_full_models = pd.concat([train, test])

train_retrain_full_models = train_retrain_full_models.dropna()
train_retrain_full_models.isna().sum()

idp_cols = SC_idp_cols()

train_retrain_full_models[idp_cols] = train_retrain_full_models[idp_cols].apply(pd.to_numeric, errors="coerce")

mean = train_retrain_full_models[idp_cols].mean()
std = train_retrain_full_models[idp_cols].std()

# Define a mask to filter out rows where any value is more than 5 SD away
mask = (train_retrain_full_models[idp_cols] - mean).abs() <= (5 * std)

# Keep only rows where all columns satisfy the condition
train_retrain_full_models = train_retrain_full_models[mask.all(axis=1)]

site_counts = train_retrain_full_models["site_id"].value_counts()

# Filter sites that have at least 10 subjects
sites_to_keep = site_counts[site_counts >= 10].index

# Keep only rows where 'site' is in the list of valid sites
train_retrain_full_models = train_retrain_full_models[train_retrain_full_models["site_id"].isin(sites_to_keep)]

#%%
ADNI_adaptation_train["which_dataset"] ="5"

train_retrain_full_models_ADNI = pd.concat([train_retrain_full_models, ADNI_adaptation_train])

train_retrain_full_models_ADNI = train_retrain_full_models_ADNI[train_retrain_full_models_ADNI['sex'].isin([0, 1])]
assert set(train_retrain_full_models_ADNI['sex']).issubset([0,1]), "Invalid values found in 'status' column"

train_retrain_full_models_ADNI["site_id"] = pd.to_numeric(
    train_retrain_full_models_ADNI["site_id"], errors="coerce"
)

missing = train_retrain_full_models_ADNI["site_id"].isna().sum()
print(f"Missing after conversion: {missing}")

train_retrain_full_models_ADNI["site_id"] = train_retrain_full_models_ADNI["site_id"].astype(int)

train_retrain_full_models_ADNI['site_id2'] = pd.factorize(train_retrain_full_models_ADNI['site_id'])[0] + 1

assert_few_nans(train_retrain_full_models_ADNI)
assert_no_negatives(train_retrain_full_models_ADNI, idp_cols)
assert_sex_column_is_binary(train_retrain_full_models_ADNI)

if write:
    train_retrain_full_models_ADNI.to_pickle(os.path.join(data_in_dir,"train_retrain_full_models_SC_adults_ADNI.pkl"))
    
    
#%% full models  - not necessary in new version
# X_train = train_retrain_full_models_ADNI[["age","sex"]].to_numpy(dtype=float)
# Z_train = train_retrain_full_models_ADNI["site_id2"].to_numpy(dtype=float)
# Y_train= train_retrain_full_models_ADNI[idp_cols].to_numpy(dtype=float)

# #%%
# # model submission Saiges models - all data used
# os.chdir(base_dir)

# if write:
#     with open('Velocity/DATA_NM/X_train_retrain_SC_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(X_train), file)  
#     with open('Velocity/DATA_NM/Y_train_retrain_SC_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_train), file)
#     with open('Velocity/DATA_NM/trbefile_retrain_SC_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Z_train), file)
        

#%% add demented files to the test set

OASIS2_full_long_demented["which_dataset"] = "4"
OASIS3_full_long_demented["which_dataset"] = "4"

cols2 = cols + ["which_dataset"]
OASIS2_full_long_demented = OASIS2_full_long_demented[cols2]
OASIS3_full_long_demented = OASIS3_full_long_demented[cols2]


ADNI_adaptation_test["which_dataset"]= "5"
ADNI_clinical["which_dataset"] ="5"

train_old = []
train_old = train
#test = pd.concat([life_te_bare, te_data_cross_non_long, data_long, OASIS2_full_long_demented, OASIS3_full_long_demented])
test = pd.concat([life_te_bare, te_data_cross_non_long, data_long, OASIS2_full_long_demented, 
                  OASIS3_full_long_demented, ADNI_adaptation_test, ADNI_clinical])

train = pd.concat([train_old,ADNI_adaptation_train ])

#%%

nan_counts = train.isna().sum()
nan_counts_test = test.isna().sum()


#%%
train = train.dropna()
test = test.dropna()

train[idp_cols] = train[idp_cols].apply(pd.to_numeric, errors="coerce")

mean = train[idp_cols].mean()
std = train[idp_cols].std()

# Define a mask to filter out rows where any value is more than 5 SD away
mask = (train[idp_cols] - mean).abs() <= (7 * std)

# Keep only rows where all columns satisfy the condition
train = train[mask.all(axis=1)]

site_counts = train["site_id"].value_counts()

# Filter sites that have at least 10 subjects
sites_to_keep = site_counts[site_counts >= 5].index

# Keep only rows where 'site' is in the list of valid sites
train = train[train["site_id"].isin(sites_to_keep)]
test = test[test["site_id"].isin(sites_to_keep)]
#%%

train_sites = train["site_id"].unique()
test_sites = test["site_id"].unique()

difference = [obj for obj in train_sites if obj not in test_sites]
print(difference)
difference = [obj for obj in test_sites if obj not in train_sites]
difference


#test4 =set(train_long["ID_subject"]) & set(test_long["ID_subject"])

#rows_to_move = train_long[train_long['ID_subject'].isin(test3)]
#train_long = train_long[~train_long['ID_subject'].isin(test3)]

#test4 =set(train_long["ID_subject"]) & set(test_long["ID_subject"])

#%%


#%%
# train.shape[0] - data_long.shape[0] + 10821 + test.shape[0]
# train_subs = train["ID_subject"].unique()
# test_subs = test["ID_subject"].unique()
# #%%
# sites = train.site_id.unique()
# for i,s in enumerate(sites):
#     idx = train['site_id'] == s
#     idxte = test['site_id'] == s
#     #print(i,s, sum(idx), sum(idxte))
#     print(sum(idx), sum(idxte))

#%%
train["set"] =1
test["set"]= 2
#%%
new = pd.concat([train, test])
#%% We wnat to do that for the full set for simplicity

new["sex"] = pd.to_numeric(
    new["sex"], errors="coerce"
)


new = new[new['sex'].isin([0, 1])]
assert set(new['sex']).issubset([0,1]), "Invalid values found in 'status' column"

new["site_id"] = pd.to_numeric(
    new["site_id"], errors="coerce"
)

missing = new["site_id"].isna().sum()
print(f"Missing after conversion: {missing}")

new["site_id"] = new["site_id"].astype(int)

new['site_id2'] = pd.factorize(new['site_id'])[0] + 1

#%%
train= new[new["set"]==1]
test= new[new["set"]==2]


#%% floatify
train["age"] = train["age"].astype(float)
train["sex"] = train["sex"].astype(float)
train["site_id2"] = train["site_id2"].astype(int)

test["age"] = test["age"].astype(float)
test["sex"] = test["sex"].astype(float)
test["site_id2"] = test["site_id2"].astype(int)

assert_few_nans(train)
assert_few_nans(test)
assert_no_negatives(train, idp_cols)
assert_no_negatives(test, idp_cols)
assert_sex_column_is_binary(train)
assert_sex_column_is_binary(test)

if write:
    train.to_pickle(os.path.join(data_in_dir,"train_SC_demented_adults_ADNI.pkl"))
    test.to_pickle(os.path.join(data_in_dir,"test_SC_demented_adults_ADNI.pkl"))

#%%

test = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/new/SC/test_SC_demented_adults_ADNI.pkl")



#%%
train_old = pd.read_pickle(os.path.join("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/new/train_SC_demented_adults_ADNI.pkl"))
test_old = pd.read_pickle(os.path.join("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/new/test_SC_demented_adults_ADNI.pkl"))


#%%
X_train = train[["age","sex"]].to_numpy(dtype=float)
Z_train = train["site_id2"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train= train[idp_cols].to_numpy(dtype=float)

X_test = test[["age","sex"]].to_numpy(dtype=float)
Z_test = test["site_id2"].to_numpy(dtype=float)
#Y_test= test[measure_name].to_numpy()
Y_test = test[idp_cols].to_numpy(dtype=float)


#%%

os.chdir(base_dir)

with open('Velocity/DATA_NM/X_train_SC_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train), file)  
with open('Velocity/DATA_NM/Y_train_SC_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train), file)
with open('Velocity/DATA_NM/trbefile_SC_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train), file)
with open('Velocity/DATA_NM/X_test_SC_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_test), file)
with open('Velocity/DATA_NM/Y_test_SC_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_test), file)
with open('Velocity/DATA_NM/tebefile_SC_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_test), file)




