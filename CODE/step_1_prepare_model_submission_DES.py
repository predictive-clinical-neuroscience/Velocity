#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:44:09 2024

@author: johbay

Destrioux
"""

#%% Create cross matched
# load 
from utils import ldpkl,  DES_idp_cols, count_site_occurrences
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from utilities_test import *
#%%

data_in_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_new_version/DES/'
write= True

#%%
base_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay'
life_tr= pd.read_csv(os.path.join(base_dir, "Lifedata_with_PING/lifespan_big_controls_extended_tr.csv"))
life_te = pd.read_csv(os.path.join(base_dir,"Lifedata_with_PING/lifespan_big_controls_extended_te.csv" ))
life_tr["site"].unique()
life_te["site"].unique()

#%%[]

UKB_full_cross = pd.read_pickle(os.path.join(data_in_dir, "UKB_cross_DES.pkl"))
IMAGEN_full_cross = pd.read_pickle(os.path.join(data_in_dir, "IMAGEN_cross_DES.pkl"))
ABCD_full_cross = pd.read_pickle(os.path.join(data_in_dir, "ABCD_cross_DES.pkl"))
OASIS2_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_cross_DES.pkl"))
OASIS3_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_cross_DES.pkl"))
#HCP_Babies_full_cross = pd.read_pickle(os.path.join(data_in_dir, "HCP_Babies_cross_DES.pkl"))
#NIH_Babies_full_cross = pd.read_pickle(os.path.join(data_in_dir,"NIH_Babies_cross_DES.pkl" )) # ad

ADNI_train = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/ADNI_DES_adaptation_train.pkl')

#%%

UKB_full_long = pd.read_pickle(os.path.join(data_in_dir, "UKB_long_DES.pkl"))
IMAGEN_full_long = pd.read_pickle(os.path.join(data_in_dir, "IMAGEN_long_DES.pkl"))
ABCD_full_long = pd.read_pickle(os.path.join(data_in_dir, "ABCD_long_DES.pkl"))
OASIS2_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_DES.pkl"))
OASIS3_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_DES.pkl"))
#HCP_Babies_full_long = pd.read_pickle(os.path.join(data_in_dir, "HCP_Babies_long_DES.pkl"))
#NIH_Babies_full_long = pd.read_pickle(os.path.join(data_in_dir, "NIH_Babies_long_DES.pkl"))

OASIS2_full_long_demented =  pd.read_pickle(os.path.join(data_in_dir, "OASIS2_demented_long_DES.pkl"))
OASIS3_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_demented_long_DES.pkl"))
ADNI_test = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/ADNI_DES_adaptation_test.pkl')
ADNI_clinical = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/ADNI_DES_clinical_goodsites.pkl')
#%% 

idp_cols = DES_idp_cols()
additional_cols = ["age","sex", "site", "site_id", "ID_visit", "ID_subject", "Mean_Thickness", "Median_Thickness"]
all_cols = idp_cols + additional_cols

#%%
life_tr["ID_visit"] = 1
life_te["ID_visit"] = 1

life_tr = life_tr.rename(columns={"Unnamed: 0":"ID"})
life_te = life_te.rename(columns={"Unnamed: 0":"ID"})

life_tr["ID_subject"] = life_tr["ID"]
life_te["ID_subject"] = life_te["ID"]

#%%

ADNI_train["site"] = "ADNI_" + str(ADNI_train["site2"])

IMAGEN_cross = IMAGEN_full_cross[all_cols]
UKB_cross = UKB_full_cross[all_cols]
ABCD_cross = ABCD_full_cross[all_cols]
OASIS2_cross = OASIS2_full_cross[all_cols]
OASIS3_cross = OASIS3_full_cross[all_cols]
#HCP_Babies_cross = HCP_Babies_full_cross[all_cols] 
#NIH_Babies_cross = NIH_Babies_full_cross[all_cols]
ADNI_train = ADNI_train[all_cols]

#%%
#data_cross = [IMAGEN_cross, OASIS2_cross, UKB_cross, ABCD_cross, OASIS3_cross, HCP_Babies_cross, NIH_Babies_cross]
data_cross = [IMAGEN_cross, OASIS2_cross, UKB_cross, ABCD_cross, OASIS3_cross, ADNI_train]
data_cross = pd.concat(data_cross)

data_cross = data_cross.dropna(how='any')
#%%
ADNI_test["site"] = "ADNI_" + str(ADNI_test["site2"])
ADNI_clinical["site"] = "ADNI_" + str(ADNI_clinical["site2"])

IMAGEN_long = IMAGEN_full_long[all_cols]
UKB_long = UKB_full_long[all_cols]
ABCD_long = ABCD_full_long[all_cols]
OASIS2_long = OASIS2_full_long[all_cols]
OASIS3_long = OASIS3_full_long[all_cols]
#HCP_Babies_long = HCP_Babies_full_long[all_cols] 
#NIH_Babies_long = NIH_Babies_full_long[all_cols]
ADNI_test = ADNI_test[all_cols]
ADNI_clinical = ADNI_clinical[all_cols]

OASIS2_long_demented = OASIS2_full_long_demented[all_cols]
OASIS3_long_demented = OASIS3_full_long_demented[all_cols]
#%%

#data_long = [IMAGEN_long, OASIS2_long, UKB_long, ABCD_long, OASIS3_long, HCP_Babies_long, NIH_Babies_long]
data_long = [IMAGEN_long, OASIS2_long, UKB_long, ABCD_long, OASIS3_long, ADNI_test]
data_long = pd.concat(data_long)

data_long = data_long.dropna(how='any')
data_long["sex"] =data_long["sex"].astype(int)

assert_few_nans(data_long)
assert_no_negatives(data_long, idp_cols)
assert_sex_column_is_binary(data_long)

assert_few_nans(data_cross)
assert_no_negatives(data_cross, idp_cols)
assert_sex_column_is_binary(data_cross)

#%%
if write:
    data_long.to_pickle(os.path.join(data_in_dir,"data_long_DES_adults.pkl"))
    data_long.to_csv(os.path.join(data_in_dir, "csv/data_long_DES_adults.csv"),index=False )
    data_cross.to_pickle(os.path.join(data_in_dir,"data_cross_DES_adults.pkl"))
    data_cross.to_csv(os.path.join(data_in_dir, "csv/data_cross_DES_adults.csv"),index=False )
 
#%% assignment to training and test data needs to be balanced for age, sex and 
# site, but also one subject can only be assigned to either training or test set (with all data points)

#%% purge the life data set
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
life_tr_bare['site_id'] = pd.factorize(life_tr_bare['site'])[0]
life_tr_bare['site_id'] = life_tr_bare['site_id'] +1
life_tr_bare['site_id'] = life_tr_bare['site_id'] +500

ref = life_tr_bare.groupby('site')['site_id'].first()
life_te_bare['site_id'] = life_te_bare['site'].map(ref)


test = life_tr_bare["site"].value_counts()
test2 = life_te_bare["site"].value_counts()
#%%
life_te_bare.columns = life_te_bare.columns.str.replace("_thickness", "", regex=True)
life_tr_bare.columns = life_tr_bare.columns.str.replace("_thickness", "", regex=True)

life_te_bare.columns = life_te_bare.columns.str.replace("&", "_and_", regex=True)
life_tr_bare.columns = life_tr_bare.columns.str.replace("&", "_and_", regex=True)

life_tr_bare["Mean_Thickness"]= life_tr_bare[idp_cols].mean(axis=1)
life_tr_bare["Median_Thickness"]= life_tr_bare[idp_cols].median(axis=1)

life_te_bare["Mean_Thickness"]= life_te_bare[idp_cols].mean(axis=1)
life_te_bare["Median_Thickness"]= life_te_bare[idp_cols].median(axis=1)


life_te_bare = life_te_bare[all_cols]
life_tr_bare = life_tr_bare[all_cols]

life_te_bare = life_te_bare[~life_te_bare["site_id"].isna()]
life_tr_bare = life_tr_bare[~life_tr_bare["site_id"].isna()]

#%%
life_tr_bare['site_id'].value_counts()
life_te_bare['site_id'].value_counts()
#%%
test=life_te_bare['site_id'].unique()
train=life_te_bare['site_id'].unique()

test - train
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

data_long.shape[0] #24591

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
train = pd.concat([life_tr_bare, tr_data_cross_non_long])
test = pd.concat([life_te_bare, te_data_cross_non_long, data_long])

test.site_id = test.site_id.astype(int)
train.site_id = train.site_id.astype(int)
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
difference
difference = [obj for obj in test_sites if obj not in train_sites]
difference

#%%
site_802 = test[test["site_id"].astype(str).str.contains("802", na=False)]
site_803 = test[test["site_id"].astype(str).str.contains("803", na=False)]
site_804 = test[test["site_id"].astype(str).str.contains("804", na=False)]
site_805 = test[test["site_id"].astype(str).str.contains("805", na=False)]

site_914 = test[test["site_id"].astype(str).str.contains("914", na=False)]
site_918 = test[test["site_id"].astype(str).str.contains("918", na=False)]
site_927 = test[test["site_id"].astype(str).str.contains("927", na=False)]
site_957 = test[test["site_id"].astype(str).str.contains("957", na=False)]
site_982 = test[test["site_id"].astype(str).str.contains("982", na=False)]
site_1000 = test[test["site_id"].astype(str).str.contains("1000", na=False)]
site_1014 = test[test["site_id"].astype(str).str.contains("1014", na=False)]
site_1033 = test[test["site_id"].astype(str).str.contains("1033", na=False)]

site_800 =  pd.concat([site_802, site_803, site_804, site_805, site_914, site_918,
                       site_927, site_957, site_982, site_1000, site_1014, site_1033])

id_list = site_800["ID_subject"].unique()
tr = rng.random(id_list.shape[0]) < 0.2
transfer = id_list[tr]

transfer_subjects = test[test["ID_subject"].isin(transfer)]

test = test[~test["ID_subject"].isin(transfer)]
train = pd.concat([train, transfer_subjects])

#%%
train_retrain_full_models = pd.concat([train, test])

train_retrain_full_models = train_retrain_full_models.dropna()
train_retrain_full_models.isna().sum()

train_retrain_full_models.shape[0] #73880

#%% remove all subjects that are more than 5 SD away from the column mean

idp_cols = DES_idp_cols()
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
#%% Check sex

print(train_retrain_full_models["sex"].dtype)
train_retrain_full_models["sex"].unique()

train_retrain_full_models = train_retrain_full_models[train_retrain_full_models['sex'].isin([0, 1])]

train_retrain_full_models["site_id"] = pd.to_numeric(
    train_retrain_full_models["site_id"], errors="coerce"
)

train_retrain_full_models["sex"] = pd.to_numeric(
    train_retrain_full_models["sex"], errors="coerce"
)

#%%


# train_retrain_full_models = train_retrain_full_models[train_retrain_full_models['site_id'].isin([501,   502,   503,   504,   505,   506,   507,   508,   509,
#          510,   511,   512,   513,   514,   515,   516,   517,   518,
#          519,   520,   521,   522,   523,   524,   525,   526,   527,
#          528,   529,   530,   531,   532,   533,   534,   535,   536,
#          537,   539,   540,   538,   541,   542,   543,   544,   545,
#          546,   548,   547,   549,   550,   551,   552,   553,   554,
#          555,   556,   558,   560,   561,   562,   563,   564,   559,
#          700,   701,   702,   703,   704,   705,   707,   706, 11027,
#        11025, 11026,   54022, 54020, 54017, 54003, 54004, 54012, 54014, 54006, 54019,
#               54016, 54002, 54005, 54001, 54010, 54013, 54011, 54009, 54007,
#               54015, 54008, 54018, 54021, 401,   402,   403, 405])]


train_retrain_full_models['site_id2'] = pd.factorize(train_retrain_full_models['site_id'])[0] + 1 
train_retrain_full_models["site_id2"].unique()

assert_few_nans(train_retrain_full_models)
assert_no_negatives(train_retrain_full_models, idp_cols)
assert_sex_column_is_binary(train_retrain_full_models)

train_retrain_full_models.shape[0]
train_retrain_full_models[
    train_retrain_full_models["which_dataset"].isin(["1","3"])
].shape[0] #48696

if write:
    train_retrain_full_models.to_pickle(os.path.join(data_in_dir,"train_retrain_full_models_DES_adults_ADNI.pkl"))   
#%% full models 
train_retrain_full_models = ldpkl(os.path.join(data_in_dir,"train_retrain_full_models_DES_adults_ADNI.pkl"))
#%%

def test_frame():
    assert train_retrain_full_models["sex"].between(0.0,1.0).all(), "Out-of-range score detected" 
    
test_frame()

#%%

X_train = train_retrain_full_models[["age","sex"]].to_numpy(dtype=float)
Z_train = train_retrain_full_models["site_id2"].to_numpy(dtype=int)
Y_train= train_retrain_full_models[idp_cols].to_numpy(dtype=float)

#%%
# model submission Saiges models - all data used
os.chdir(base_dir)

with open('Velocity/DATA_NM/X_train_retrain_DES_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train), file)  
with open('Velocity/DATA_NM/Y_train_retrain_DES_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train), file)
with open('Velocity/DATA_NM/trbefile_retrain_DES_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train), file)

#%%
#train = pd.concat([life_tr_bare, tr_data_cross_non_long])
#test = pd.concat([life_te_bare, te_data_cross_non_long, data_long])

OASIS2_full_long_demented["which_dataset"] = "4"
OASIS3_full_long_demented["which_dataset"] = "4"

ADNI_clinical["which_dataset"] = "5"
 
cols2 = all_cols + ["which_dataset"]
OASIS2_full_long_demented = OASIS2_full_long_demented[cols2]
OASIS3_full_long_demented = OASIS3_full_long_demented[cols2]

test = pd.concat([test, OASIS2_full_long_demented, OASIS3_full_long_demented, ADNI_clinical])

test.site_id = test.site_id.astype(int)
train.site_id = train.site_id.astype(int)

#%%
idp_cols = DES_idp_cols()
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
test = test[test["site_id"].isin(sites_to_keep)]

train_sites = train["site_id"].unique()
test_sites = test["site_id"].unique()

difference = [obj for obj in train_sites if obj not in test_sites]
difference
difference = [obj for obj in test_sites if obj not in train_sites]
difference


#%%
train = train.dropna()
test = test.dropna()

#test4 =set(train_long["ID_subject"]) & set(test_long["ID_subject"])

#rows_to_move = train_long[train_long['ID_subject'].isin(test3)]
#train_long = train_long[~train_long['ID_subject'].isin(test3)]

#test4 =set(train_long["ID_subject"]) & set(test_long["ID_subject"])

#remove outlier sex values
train['sex'] = train['sex'].astype(float)
print(train["sex"].dtype)
train["sex"].unique()

test['sex'] = test['sex'].astype(float)
print(test["sex"].dtype)
test["sex"].unique()

train = train[train['sex'].isin([0, 1])]
test = test[test['sex'].isin([0, 1])]

#%%

#%%
sites = train.site_id.unique()
for i,s in enumerate(sites):
    idx = train['site_id'] == s
    idxte = test['site_id'] == s
    #print(i,s, sum(idx), sum(idxte))
    print(sum(idx), sum(idxte))

#%%
count_site_occurrences(train_retrain_full_models)

#%% Something was wrong with the site ids. We have to extract them and re-number them
#
new_train = train
new_train["set"] = 1
#%%
new_test = test
new_test["set"] = 2
#%%
new = pd.concat([new_train, new_test])
#%% We wnat to do that for the full set for simplicity

new['site_id2'] = pd.factorize(new['site_id'])[0] + 1
#%%
train= new[new["set"]==1]
test= new[new["set"]==2]

assert_few_nans(train)
assert_few_nans(test)
assert_no_negatives(train, idp_cols)
assert_no_negatives(test, idp_cols)
assert_sex_column_is_binary(train)
assert_sex_column_is_binary(test)

train.shape[0] #28104
test.shape[0] #46382


if write:
    train.to_pickle(os.path.join(data_in_dir,"train_DES_demented_adults_ADNI.pkl"))
    test.to_pickle(os.path.join(data_in_dir,"test_DES_demented_adults_ADNI.pkl"))

#%%
X_train = train[["age","sex"]].to_numpy(dtype=float)
Z_train = train["site_id2"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train= train[idp_cols].to_numpy(dtype=float)

X_test = test[["age","sex"]].to_numpy(dtype=float)
Z_test = test["site_id2"].to_numpy(dtype=float)
#Y_test= test[measure_name].to_numpy()
Y_test= test[idp_cols].to_numpy(dtype=float)

#%%

os.chdir(base_dir)

with open('Velocity/DATA_NM/X_train_DES_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train), file)  
with open('Velocity/DATA_NM/Y_train_DES_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train), file)
with open('Velocity/DATA_NM/trbefile_DES_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train), file)
with open('Velocity/DATA_NM/X_test_DES_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_test), file)
with open('Velocity/DATA_NM/Y_test_DES_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_test), file)
with open('Velocity/DATA_NM/tebefile_DES_demented_adults_ADNI.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_test), file)




#%%
import matplotlib.pyplot as plt
import pandas as pd
site_ids = np.unique(train["site_id"])

# Loop through each site and plot
for site_id in site_ids:
    site_data = train[train["site_id"] == site_id]
    x = site_data["age"]
    y = site_data.iloc[:, 0]  # column 0
    plt.scatter(x, y, alpha=0.6)

    plt.xlabel("Age")
    plt.ylabel("Column 0 values")
    plt.title(f"Column 0 vs Age, site_id = {site_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, 100)     # Set x-axis from 0 to 100
    plt.ylim(0, 4) 
    plt.show()
    
#%%
data_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/'

X_train_path = os.path.join(data_dir, 'X_train_DES_demented_adults.pkl')
Y_train_path = os.path.join(data_dir, 'Y_train_DES_demented_adults.pkl')
Z_train_path = os.path.join(data_dir, 'trbefile_DES_demented_adults.pkl')

X_test_path = os.path.join(data_dir, 'X_test_DES_demented_adults.pkl')
Y_test_path = os.path.join(data_dir, 'Y_test_DES_demented_adults.pkl')
Z_test_path = os.path.join(data_dir, 'tebefile_DES_demented_adults.pkl')

Z_train = pd.read_pickle(Z_train_path)

#%%
os.chdir(base_dir)

train_1 = train[train["which_dataset"] == "1"]

X_train_1 = train_1[["age","sex"]].to_numpy(dtype=float)
Z_train_1 = train_1["site_id"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train_1= train_1[idp_cols].to_numpy(dtype=float)

#X_test = test[["age","sex"]].to_numpy(dtype=float)
#Z_test = test["site_id"].to_numpy(dtype=float)
#Y_test= test[measure_name].to_numpy()
#Y_test= test[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_train_DES_1.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train_1), file)  
with open('Velocity/DATA_NM/Y_train_DES_1.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train_1), file)
with open('Velocity/DATA_NM/trbefile_DES_1.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train_1), file)
#%%
os.chdir(base_dir)
idp_cols = idp_cols[0:15]

train_3 = train[train["which_dataset"] == "3"]

X_train_3 = train_3[["age","sex"]].to_numpy(dtype=float)
Z_train_3 = train_3["site_id"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train_3= train_3[idp_cols].to_numpy(dtype=float)

#X_test = test[["age","sex"]].to_numpy(dtype=float)
#Z_test = test["site_id"].to_numpy(dtype=float)
#Y_test= test[measure_name].to_numpy()
#Y_test= test[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_train_DES_3.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train_3), file)  
with open('Velocity/DATA_NM/Y_train_DES_3.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train_3), file)
with open('Velocity/DATA_NM/trbefile_DES_3.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train_3), file)

#%%
os.chdir(base_dir)
idp_cols = idp_cols[0:15]

test_2 = test[test["which_dataset"] == "2"]

X_test_2 = test_2[["age","sex"]].to_numpy(dtype=float)
Z_test_2 = test_2["site_id"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_test_2= test_2[idp_cols].to_numpy(dtype=float)

#X_test = test[["age","sex"]].to_numpy(dtype=float)
#Z_test = test["site_id"].to_numpy(dtype=float)
#Y_test= test[measure_name].to_numpy()
#Y_test= test[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_test_DES_2.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_test_2), file)  
with open('Velocity/DATA_NM/Y_test_DES_2.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_test_2), file)
with open('Velocity/DATA_NM/tebefile_DES_2.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_test_2), file)

#%%
os.chdir(base_dir)
idp_cols = idp_cols[0:15]
UKB_train = train[train['site_id'].isin([11027, 11025, 11026])]

X_train_UKB = UKB_train[["age","sex"]].to_numpy(dtype=float)
Z_train_UKB = UKB_train["site_id"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train_UKB= UKB_train[idp_cols].to_numpy(dtype=float)

#X_test = test[["age","sex"]].to_numpy(dtype=float)
#Z_test = test["site_id"].to_numpy(dtype=float)
#Y_test= test[measure_name].to_numpy()
#Y_test= test[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_train_DES_UKB.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train_UKB), file)  
with open('Velocity/DATA_NM/Y_train_DES_UKB.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train_UKB), file)
with open('Velocity/DATA_NM/trbefile_DES_UKB.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train_UKB), file)
    
#%%
os.chdir(base_dir)
train_1_UKB = pd.concat([train_1, UKB_train])

X_train_1_UKB = train_1_UKB[["age","sex"]].to_numpy(dtype=float)
Z_train_1_UKB = train_1_UKB["site_id"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train_1_UKB= train_1_UKB[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_train_DES_UKB_1.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train_1_UKB), file)  
with open('Velocity/DATA_NM/Y_train_DES_UKB_1.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train_1_UKB), file)
with open('Velocity/DATA_NM/trbefile_DES_UKB_1.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train_1_UKB), file)

#%%
os.chdir(base_dir)
idp_cols = idp_cols[0:15]
ABCD_train = train[train['site_id'].isin([54022, 54020, 54017, 54003, 54004, 54012, 54014,
54006, 54019, 54016, 54002, 54005, 54001, 54010, 54013, 54011,
54009, 54007, 54015, 54008, 54018, 54021])]

ABCD_test = test[test['site_id'].isin([54022, 54020, 54017, 54003, 54004, 54012, 54014,
54006, 54019, 54016, 54002, 54005, 54001, 54010, 54013, 54011,
54009, 54007, 54015, 54008, 54018, 54021])]

X_train_ABCD = ABCD_train[["age","sex"]].to_numpy(dtype=float)
Z_train_ABCD = ABCD_train["site_id"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train_ABCD= ABCD_train[idp_cols].to_numpy(dtype=float)

X_test_ABCD = ABCD_test[["age","sex"]].to_numpy(dtype=float)
Z_test_ABCD = ABCD_test["site_id"].to_numpy(dtype=float)
Y_test_ABCD= ABCD_test[idp_cols].to_numpy(dtype=float)

df_big = pd.read_csv("/home/common/temporary/4Johanna/df_big.csv")

df_big_sub = df_big[df_big['site'].str.contains("ABCD", na=False)]


with open('Velocity/DATA_NM/X_train_DES_ABCD.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train_ABCD), file)  
with open('Velocity/DATA_NM/Y_train_DES_ABCD.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train_ABCD), file)
with open('Velocity/DATA_NM/trbefile_DES_ABCD.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train_ABCD), file)

#%%
os.chdir(base_dir)
train_1_UKB_ABCD = pd.concat([train_1, UKB_train, ABCD_train])

X_train_1_UKB_ABCD = train_1_UKB_ABCD[["age","sex"]].to_numpy(dtype=float)
Z_train_1_UKB_ABCD = train_1_UKB_ABCD["site_id"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train_1_UKB_ABCD= train_1_UKB_ABCD[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_train_DES_UKB_1_ABCD.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train_1_UKB_ABCD), file)  
with open('Velocity/DATA_NM/Y_train_DES_UKB_1_ABCD.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train_1_UKB_ABCD), file)
with open('Velocity/DATA_NM/trbefile_DES_UKB_1_ABCD.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train_1_UKB_ABCD), file)

#%%
train3 = train[train['site_id'].isin([501,   502,   503,   504,   505,   506,   507,   508,   509,
         510,   511,   512,   513,   514,   515,   516,   517,   518,
         519,   520,   521,   522,   523,   524,   525,   526,   527,
         528,   529,   530,   531,   532,   533,   534,   535,   536,
         537,   539,   540,   538,   541,   542,   543,   544,   545,
         546,   548,   547,   549,   550,   551,   552,   553,   554,
         555,   556,   558,   560,   561,   562,   563,   564,   559,
         700,   701,   702,   703,   704,   705,   707,   706, 11027,
       11025, 11026,   54022, 54020, 54017, 54003, 54004, 54012, 54014, 54006, 54019,
              54016, 54002, 54005, 54001, 54010, 54013, 54011, 54009, 54007,
              54015, 54008, 54018, 54021, 401,   402,   403,
         405])]

train3['site_id2'] = pd.factorize(train3['site_id'])[0] + 1

os.chdir(base_dir)

X_train3 = train3[["age","sex"]].to_numpy(dtype=float)
Z_train3 = train3["site_id2"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train3= train3[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_train3_DES.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train3), file)  
with open('Velocity/DATA_NM/Y_train3_DES.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train3), file)
with open('Velocity/DATA_NM/trbefile3_DES.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train3), file)

#%%
os.chdir(base_dir)

X_train2 = train2[["age","sex"]].to_numpy(dtype=float)
Z_train2 = train2["site_id2"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train2= train2[idp_cols].to_numpy(dtype=float)

with open('Velocity/DATA_NM/X_train2_DES.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train2), file)  
with open('Velocity/DATA_NM/Y_train2_DES.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train2), file)
with open('Velocity/DATA_NM/trbefile2_DES.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train2), file)
