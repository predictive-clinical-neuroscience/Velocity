#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:42:18 2024

@author: johbay
"""

#%%
"""
This script balances longitudianl and cross sectional data, DK atlas
"""


# load 
from utils import ldpkl, DK_idp_cols
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pandas.testing import assert_frame_equal
from utilities_test import *

#%%

data_in_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_new_version/DK'
write= False
#%%
base_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay'
life_tr= pd.read_csv(os.path.join(base_dir, "Lifedata_with_PING/DK_extended_lifespan_baby_big_ct_tr.csv"))
life_te = pd.read_csv(os.path.join(base_dir,"Lifedata_with_PING/DK_extended_lifespan_baby_big_ct_te.csv" ))
life_tr["site"].unique()
life_te["site"].unique()

data_in_dir_ADNI = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/'
#%%[]

UKB_full_cross = pd.read_pickle(os.path.join(data_in_dir, "UKB_cross_DK.pkl"))
IMAGEN_full_cross = pd.read_pickle(os.path.join(data_in_dir, "IMAGEN_cross_DK.pkl"))
ABCD_full_cross = pd.read_pickle(os.path.join(data_in_dir, "ABCD_cross_DK.pkl"))
OASIS2_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_cross_DK.pkl"))
OASIS3_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_cross_DK.pkl"))
#HCP_Babies_full_cross = pd.read_pickle(os.path.join(data_in_dir, "HCP_Babies_cross_DK.pkl"))
#NIH_Babies_full_cross = pd.read_pickle(os.path.join(data_in_dir,"NIH_Babies_cross_DK.pkl" )) # ad

ADNI_train = pd.read_pickle(os.path.join(data_in_dir_ADNI, "ADNI_DK_adaptation_train.pkl"))
ADNI_test = pd.read_pickle(os.path.join(data_in_dir_ADNI, "ADNI_DK_adaptation_test.pkl"))
ADNI_clinical = pd.read_pickle(os.path.join(data_in_dir_ADNI, "ADNI_DK_clinical_goodsites.pkl"))
#%%

UKB_full_long = pd.read_pickle(os.path.join(data_in_dir, "UKB_long_DK.pkl"))
IMAGEN_full_long = pd.read_pickle(os.path.join(data_in_dir, "IMAGEN_long_DK.pkl"))
ABCD_full_long = pd.read_pickle(os.path.join(data_in_dir, "ABCD_long_DK.pkl"))
OASIS2_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_DK.pkl"))
OASIS3_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_DK.pkl"))
#HCP_Babies_full_long = pd.read_pickle(os.path.join(data_in_dir, "HCP_Babies_long_DK.pkl"))
#NIH_Babies_full_long = pd.read_pickle(os.path.join(data_in_dir, "NIH_Babies_long_DK.pkl"))

OASIS2_full_long_demented =  pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_DK_demented.pkl"))
OASIS3_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_DK_demented.pkl"))
#%% 

idp_cols = DK_idp_cols()
additional_cols = ["age","sex", "site_id", "ID_visit", "ID_subject", "Mean_Thickness", "Median_Thickness"]
all_cols = idp_cols + additional_cols

#%%
life_tr["ID_visit"] = 1
life_te["ID_visit"] = 1

life_tr = life_tr.rename(columns={"Unnamed: 0":"ID"})
life_te = life_te.rename(columns={"Unnamed: 0":"ID"})

life_tr["ID_subject"] = life_tr["ID"]
life_te["ID_subject"] = life_te["ID"]

assert_no_negatives(life_tr, idp_cols)
assert_no_negatives(life_te, idp_cols)

assert_sex_column_is_binary(life_te)
assert_sex_column_is_binary(life_tr)

assert_few_nans(life_te)
assert_few_nans(life_tr)
#%%

IMAGEN_cross = IMAGEN_full_cross[all_cols]
UKB_cross = UKB_full_cross[all_cols]
ABCD_cross = ABCD_full_cross[all_cols]
OASIS2_cross = OASIS2_full_cross[all_cols]
OASIS3_cross = OASIS3_full_cross[all_cols]
#HCP_Babies_cross = HCP_Babies_full_cross[all_cols] 
#NIH_Babies_cross = NIH_Babies_full_cross[all_cols]

ADNI_train = ADNI_train[all_cols]
ADNI_test = ADNI_test[all_cols]
ADNI_clinical = ADNI_clinical[all_cols]
#%%
data_cross = [IMAGEN_cross, OASIS2_cross, UKB_cross, ABCD_cross, OASIS3_cross, ADNI_train]
#data_cross = [IMAGEN_cross, OASIS2_cross, UKB_cross, ABCD_cross, OASIS3_cross, HCP_Babies_cross, NIH_Babies_cross]

data_cross = pd.concat(data_cross)

data_cross = data_cross.dropna(how='any')
#%%

IMAGEN_long = IMAGEN_full_long[all_cols]
UKB_long = UKB_full_long[all_cols]
ABCD_long = ABCD_full_long[all_cols]
OASIS2_long = OASIS2_full_long[all_cols]
OASIS3_long = OASIS3_full_long[all_cols]
#HCP_Babies_long = HCP_Babies_full_long[all_cols] 
#NIH_Babies_long = NIH_Babies_full_long[all_cols]

OASIS2_long_demented = OASIS2_full_long_demented[all_cols]
OASIS3_long_demented = OASIS3_full_long_demented[all_cols]
#%%

#data_long = [IMAGEN_long, OASIS2_long, UKB_long, ABCD_long, OASIS3_long, HCP_Babies_long, NIH_Babies_long]
data_long = [IMAGEN_long, OASIS2_long, UKB_long, ABCD_long, OASIS3_long, ADNI_test]

data_long = pd.concat(data_long)

data_long = data_long.dropna(how='any')

assert_few_nans(data_long)
assert_no_negatives(data_long, idp_cols)
assert_sex_column_is_binary(data_long)

assert_few_nans(data_cross)
assert_no_negatives(data_cross, idp_cols)
assert_sex_column_is_binary(data_cross)

#%%
if write:
    data_long.to_pickle(os.path.join(data_in_dir,"data_long_DK_ADNI.pkl"))
    data_long.to_csv(os.path.join(data_in_dir, "csv/data_long_DK_ADNI.csv"),index=False )
    data_cross.to_pickle(os.path.join(data_in_dir,"data_cross_DK_ADNI.pkl"))
    data_cross.to_csv(os.path.join(data_in_dir, "csv/data_cross_DK_ADNI.csv"),index=False )

#%%
# purge the life data set
drop_sites =['ukb-11027.0', 'ukb-11025.0', 'ukb-11026.0', 'Oasis3_2',
       'Oasis3_1', 'Oasis3_4', 'Oasis3_5', 'Oasis3_3','ABCD_50010.0', 'ABCD_50002.0',
       'ABCD_50018.0', 'ABCD_50001.0', 'ABCD_50009.0', 'ABCD_50011.0',
       'ABCD_50004.0', 'ABCD_50025.0', 'ABCD_50014.0', 'ABCD_50021.0',
       'ABCD_50006.0', 'ABCD_50007.0', 'ABCD_50019.0', 'ABCD_50003.0',
       'ABCD_50015.0', 'ABCD_50008.0', 'ABCD_50016.0', 'ABCD_50027.0',
       'ABCD_50005.0', 'ABCD_50026.0', 'ABCD_50020.0', 'ABCD_50013.0',
       'ABCD_50022.0', 'ABCD_50012.0', 'ABCD_50017.0', 'ABCD_50024.0',
       'ABCD_50023.0', 'ABCD_50029.0', 'ABCD_50028.0', 'HCP_Baby_UMN', 'HCP_Baby_UNC', 'NIH_pedMRI_2',
       'NIH_pedMRI_3', 'NIH_pedMRI_6', 'NIH_pedMRI_1', 'NIH_pedMRI_4',
       'NIH_pedMRI_5']

#%%
life_tr_bare=life_tr[~life_tr["site"].str.contains("|".join(drop_sites))]
life_te_bare=life_te[~life_te["site"].str.contains("|".join(drop_sites))]
#%%
life_te_bare = life_te_bare[all_cols]
life_tr_bare = life_tr_bare[all_cols]

life_te_bare = life_te_bare[~life_te_bare["site_id"].isna()]
life_tr_bare = life_tr_bare[~life_tr_bare["site_id"].isna()]

#%%
life_tr_bare['site_id'].value_counts()
life_te_bare['site_id'].value_counts()

#%%
life_tr_bare["which_dataset"] = "1"
life_te_bare["which_dataset"] = "1"
#%% Longitudinal data set
#%%
# split the longitudinal data set
data_long.groupby(by="site_id").count()

data_long["ID_visit"].unique()
data_long['ID_visit'] = data_long["ID_visit"].astype(str)
data_long['ID_visit'] = data_long['ID_visit'].str.replace('v','')
data_long['ID_visit'] = data_long["ID_visit"].astype(float)
#%%
data_long["ID_subject"] = data_long["ID_subject"].astype(str)

#%%
subs_long=data_long["ID_subject"].unique()


# rng = np.random.default_rng(seed=42)
# tr = rng.random(subs_long.shape[0]) > 0.80

# te = ~tr

# subs_tr = subs_long[tr]
# subs_te = subs_long[te]

#%%
# subs_tr = subs_tr.astype(str)
# data_long["ID_subject"] = data_long["ID_subject"].astype(str)

# tr_bool = data_long["ID_subject"].str.contains("|".join(subs_tr))

# subs_te = subs_te.astype(str)
# te_bool = data_long["ID_subject"].str.contains("|".join(subs_te))

# train_long = data_long[tr_bool]
# test_long = data_long[te_bool]

#%%
data_long["which_dataset"]="2"
#train_long["which_dataset"] = "2"
#test_long["which_dataset"] = "2"

#%% remove longitudinal subjects from the cross sectional data set
data_cross["ID_subject"] = data_cross["ID_subject"].astype(str)

bool_cross = data_cross["ID_subject"].str.contains("|".join(subs_long))
#te_bool_cross = data_cross["ID_subject"].str.contains("|".join(subs_te))
bool_non_cross = ~bool_cross

data_cross_non_long = data_cross[bool_non_cross] 

#%%
data_cross_non_long["ID_subject"] = data_cross_non_long["ID_subject"].astype(str)
data_cross_non_long = data_cross_non_long.dropna(how='any')

cross_subs = data_cross_non_long["ID_subject"]
#cross_subs["ID_subject"] = cross_subs["ID_subject"].astype(str)

rng = np.random.default_rng(seed=42)
tr_cross = rng.random(cross_subs.shape[0]) > 0.4
te_cross = ~tr_cross

tr_data_cross_non_long = data_cross_non_long[tr_cross]
te_data_cross_non_long = data_cross_non_long[te_cross]

#%%
tr_data_cross_non_long["which_dataset"] = "3"
te_data_cross_non_long["which_dataset"] = "3"

#%%
OASIS2_long_demented["which_dataset"]="4"
OASIS3_long_demented["which_dataset"]="4"
ADNI_clinical["which_dataset"]="5"
#%%
train = pd.concat([life_tr_bare, tr_data_cross_non_long])
test = pd.concat([life_te_bare, te_data_cross_non_long, data_long])


#%%
test3 =set(data_long["ID_subject"]) & set(tr_data_cross_non_long["ID_subject"])

#make sure that all the sites in the training set are also in the test set
train_sites = train['site_id'].value_counts()
test_sites = test['site_id'].value_counts()


train["site_id"] = train["site_id"].astype(int)
test["site_id"] = test["site_id"].astype(int)

train["site_id"] = train["site_id"].astype(str)
test["site_id"] = test["site_id"].astype(str)


train_sites = train["site_id"].unique()
test_sites = test["site_id"].unique()

train = train.drop_duplicates()
test = test.drop_duplicates()

difference = [obj for obj in train_sites if obj not in test_sites]
print(difference)
difference = [obj for obj in test_sites if obj not in train_sites]
print(difference)


#%%

# List of site IDs you want to sample from
sites =  [str(s) for s in difference]

# How much to sample (e.g. 10%)
sample_fraction = 0.2

# Optional: for reproducibility
rng = np.random.default_rng(seed=42)

# Loop through each site
for site in sites:
    site_df = test[test["site_id"].astype(str).str.contains(site, na=False)]
    
    id_list = site_df["ID_subject"].unique()
    tr = rng.random(id_list.shape[0]) < sample_fraction
    transfer = id_list[tr]
    
    transfer_subjects = test[test["ID_subject"].isin(transfer)]
    
    test = test[~test["ID_subject"].isin(transfer)]
    train = pd.concat([train, transfer_subjects], ignore_index=True)



train_sites = train["site_id"].unique()
test_sites = test["site_id"].unique()

difference = [obj for obj in train_sites if obj not in test_sites]
print(difference)
difference = [obj for obj in test_sites if obj not in train_sites]
print(difference)

remove_sites = ['916','1014', '957', '920', '919', '912', '1053', '914']

train = train[~train["site_id"].isin(remove_sites)]
test = test[~test["site_id"].isin(remove_sites)]
ADNI_clinical = ADNI_clinical[~ADNI_clinical["site_id"].isin(remove_sites)]
#%% Write 

#train_loaded = pd.read_pickle(os.path.join(data_in_dir,"train_DK_adults.pkl"))

#assert_frame_equal(train, train_loaded)

assert_few_nans(train)
assert_no_negatives(train, idp_cols)
assert_sex_column_is_binary(train)

assert_few_nans(test)
assert_no_negatives(test, idp_cols)
assert_sex_column_is_binary(test)



# if write:
#     train.to_pickle(os.path.join(data_in_dir,"train_DK_adults.pkl"))
#     train.to_csv(os.path.join(data_in_dir, "csv/train_DK_adults.csv"),index=False )
#     test.to_pickle(os.path.join(data_in_dir,"test_DK_adults.pkl"))
#     test.to_csv(os.path.join(data_in_dir, "csv/test_DK_adults.csv"),index=False )

#%%
measures_name = idp_cols + ["Mean_Thickness", "Median_Thickness"]


#%% write the models that will be fully trained

train_retrain_full_models = pd.concat([train, test])

train_retrain_full_models = train_retrain_full_models.dropna()
train_retrain_full_models.isna().sum()

idp_cols = DK_idp_cols()
idp_cols.extend(["Mean_Thickness","Median_Thickness"])

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

#%% remove funky sexes
train_retrain_full_models['sex'] = train_retrain_full_models['sex'].astype(float)
print(train_retrain_full_models["sex"].dtype)
train_retrain_full_models["sex"].unique()


train_retrain_full_models = train_retrain_full_models[train_retrain_full_models['sex'].isin([0, 1])]
assert set(train_retrain_full_models['sex']).issubset([0,1]), "Invalid values found in 'status' column"

train_retrain_full_models["site_id"] = pd.to_numeric(
    train_retrain_full_models["site_id"], errors="coerce"
)

missing = train_retrain_full_models["site_id"].isna().sum()
print(f"Missing after conversion: {missing}")

train_retrain_full_models["site_id"] = train_retrain_full_models["site_id"].astype(int)

train_retrain_full_models['site_id2'] = pd.factorize(train_retrain_full_models['site_id'])[0] + 1

if write:
    train_retrain_full_models.to_pickle(os.path.join(data_in_dir,"train_retrain_full_models_DK_adults_ADNI.pkl"))


#%% old version

# X_train = train_retrain_full_models[["age","sex"]].to_numpy(dtype=float)
# Z_train = train_retrain_full_models["site_id2"].to_numpy(dtype=float)
# Y_train= train_retrain_full_models[measures_name].to_numpy(dtype=float)

# #%%
# # model submission Saiges models - all data used
# os.chdir(base_dir)

# if write: 
#     with open('Velocity/DATA_NM/new/X_train_retrain_DK_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(X_train), file)  
#     with open('Velocity/DATA_NM/new/Y_train_retrain_DK_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_train), file)
#     with open('Velocity/DATA_NM/new/trbefile_retrain_DK_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Z_train), file)
    

#%%
train = pd.concat([life_tr_bare, tr_data_cross_non_long])
#test= pd.read_pickle(os.path.join(data_in_dir,"test_DK_adults.pkl"))

test_demented = pd.concat([test, OASIS2_long_demented, OASIS3_long_demented, ADNI_clinical])


idp_cols = DK_idp_cols()
idp_cols.extend(["Mean_Thickness","Median_Thickness"])

train[idp_cols] = train[idp_cols].apply(pd.to_numeric, errors="coerce")

mean = train[idp_cols].mean()
std = train[idp_cols].std()

# Define a mask to filter out rows where any value is more than 5 SD away
mask = (train[idp_cols] - mean).abs() <= (5 * std)

# Keep only rows where all columns satisfy the condition
train = train[mask.all(axis=1)]

site_counts = train["site_id"].value_counts()

# Filter sites that have at least 10 subjects
sites_to_keep = site_counts[site_counts >= 5].index

# Keep only rows where 'site' is in the list of valid sites
train = train[train["site_id"].isin(sites_to_keep)]
test_demented = test_demented[test["site_id"].isin(sites_to_keep)]

train_sites = train["site_id"].unique()
test_demented_sites = test_demented["site_id"].unique()

difference = [obj for obj in train_sites if obj not in test_demented_sites]
print(difference)
difference = [obj for obj in test_demented_sites if obj not in train_sites]
print(difference)


#%%
train["set"] =1
test["set"]= 2
#%%
new = pd.concat([train, test])
#%% We wnat to do that for the full set for simplicity

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
#%%
train = train.dropna(how='any')
test = test.dropna(how='any')

# #%% This is no obsolete
# test_all.reset_index(drop = True, inplace = True)
# site_804 = test_all[test_all["site_id"]==804]
# site_805 = test_all[test_all["site_id"]==805]

# new = test_all.iloc[[18971, 18963]]  
# train_all = pd.concat([train_all, new], axis=0, ignore_index=True)

# test_all = test_all.drop([18971, 18963])

# #%%\
# new = test_all.iloc[[18871, 18877, 18882, 18887, 18902]]  
# train_all = pd.concat([train_all, new], axis=0, ignore_index=True)

# test_all = test_all.drop([18871, 18877, 18882, 18887, 18902])
train['sex'] = train['sex'].astype(float)
print(train["sex"].dtype)
train["sex"].unique()

test['sex'] = test['sex'].astype(float)
print(test["sex"].dtype)
test["sex"].unique()

train = train[train['sex'].isin([0, 1])]
test = test[test['sex'].isin([0, 1])]


if write:
     train.to_pickle(os.path.join(data_in_dir,"train_DK_demented_adults_ADNI.pkl"))
     train.to_csv(os.path.join(data_in_dir, "csv/train_DK_demtented_adults_ADNI.csv"),index=False )
     test.to_pickle(os.path.join(data_in_dir,"test_DK_demented_adults_ADNI.pkl"))
     test.to_csv(os.path.join(data_in_dir, "csv/test_DK_demented_adults_ADNI.csv"),index=False )


    
#%% old version

# X_train = train[["age", "sex"]].to_numpy(dtype=float)
# Z_train = train["site_id2"].to_numpy(dtype=float)
# Y_train = train[measures_name].to_numpy(dtype=float)

# X_test = test[["age", "sex"]].to_numpy(dtype=float)
# Z_test = test["site_id2"].to_numpy(dtype=float)
# Y_test = test[measures_name].to_numpy(dtype=float)

# #%%
# os.chdir('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/')

# if write: 
#     with open('Velocity/DATA_NM/new/X_train_DK_demented_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(X_train), file)  
#     with open('Velocity/DATA_NM/new/Y_train_DK_demented_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_train), file)
#     with open('Velocity/DATA_NM/new/trbefile_DK_demented_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Z_train), file)
#     with open('Velocity/DATA_NM/new/X_test_DK_demented_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(X_test), file)
#     with open('Velocity/DATA_NM/new/Y_test_DK_demented_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Y_test), file)
#     with open('Velocity/DATA_NM/new/tebefile_DK_demented_adults_ADNI.pkl', 'wb') as file:
#         pickle.dump(pd.DataFrame(Z_test), file)
