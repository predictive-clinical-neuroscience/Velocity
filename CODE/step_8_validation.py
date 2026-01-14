#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:19:07 2025

@author: johbay
"""

import pandas as pd
import pcntoolkit as pcn 
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import SC_idp_cols, DK_idp_cols
import seaborn as sns
from scipy.sparse import diags
#%%
idp_cols = SC_idp_cols()
data_in_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed'
base_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/'

#%% load demented files

OASIS2_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_cross_SC.pkl"))
OASIS2_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_SC.pkl"))

OASIS3_full_long = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_SC.pkl"))
OASIS3_full_cross = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_cross_SC.pkl"))

OASIS2_full_cross_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_cross_SC_demented.pkl"))
OASIS2_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_SC_demented.pkl"))

OASIS3_full_cross_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_cross_SC_demented.pkl"))
OASIS3_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_SC_demented.pkl"))

#%% create OASIS data sets 

OASIS_long_demented = pd.concat([OASIS2_full_long_demented, OASIS3_full_long_demented], axis =0)
OASIS_long = pd.concat([OASIS2_full_long, OASIS3_full_long], axis =0)

#%% look at difference between demented and non-demented 
for _, group_data in OASIS2_full_long.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color='black', alpha=0.3)
for group_name, group_data in OASIS2_full_long_demented.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color="red", alpha = 0.7)



# for _, group_data in OASIS3_full_long.groupby('ID_subject'):
#     plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color='black', alpha=0.3)
# for group_name, group_data in OASIS3_full_long_demented.groupby('ID_subject'):
#     plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color="red", alpha = 0.7)

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
#all OASIS
for _, group_data in OASIS_long.groupby('ID_subject'):
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color='black', alpha=0.3)

# Now: plot demented subjects in color by diagnosis
diagnosis_colors = {
    2.0: 'red',
    1.0: 'blue'
    # Add more if needed
}

for _, group_data in OASIS_long_demented.groupby('ID_subject'):
    diagnosis = group_data['diagnosis'].iloc[0]  # or 'Diagnosis' depending on your column name
    color = diagnosis_colors.get(diagnosis, 'blue')  # default to blue if not found
    plt.plot(group_data['age'], group_data['Left-Lateral-Ventricle'], color=color, alpha=0.7)
    
plt.show()

#%% OASIS in z-scores / Left Lateral ventricle
measures = SC_idp_cols()
model_type ="SHASHb_1"
atlas = "SC"
idp_nr =1
measure = measures[idp_nr]

base_dir= ('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/')
data_dir = os.path.join(base_dir, "Velocity/DATA_pre_processed")
test_raw = pd.read_pickle(
    '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/test_SC_demented_adults.pkl')

batch_dir = f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/{model_type}_estimate_scaled_fixed_{atlas}_demented_adults/batch_{idp_nr +1}/'

test_raw["age"] = test_raw["age"].astype(float)
test_raw["ID_subject"] = test_raw["ID_subject"].astype(str)
test_raw["sex"] = test_raw["sex"].astype(str)
test_raw["site_id"] = test_raw["site_id"].astype(str)
test_raw["ID_visit"].unique()
test_raw["ID_visit"] = test_raw["ID_visit"].astype(str)
test_raw["ID_visit"] = test_raw["ID_visit"].str.replace('v', '')
test_raw["ID_visit"] = test_raw["ID_visit"].astype(float)
test_raw["which_dataset"] = test_raw["which_dataset"].astype(int)
test_raw.reset_index(drop=True, inplace=True)
#
#  get the z-scores for
Z_measures = pd.read_pickle(os.path.join(batch_dir, 'Z_estimate.pkl'))

test_raw.reset_index(drop=True, inplace=True)

# add covarites
data_all = pd.concat(
    [Z_measures, test_raw[["ID_subject", "ID_visit", "age", "sex", "site_id", "which_dataset"]]],  axis=1)

data_all.rename(columns={data_all.columns[0]: measure}, inplace=True )

#
# OAS2_0157 is a good example
#OASIS2_demented = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Longitudinal/Data/OASIS2_sc_demented.pkl")

data = data_all[data_all["which_dataset"]==2]

data["age"] = data["age"].round()
data["age"] =  data["age"].astype(int)
    
data["ID_visit"].astype(int)
data["ID_subject"].astype(str)


data_demented = data_all[data_all["which_dataset"]==4]

#%%
data.loc[data['site_id'].astype(str).str.startswith('540'), 'site'] = 'ABCD'
data.loc[data['site_id'].astype(str).str.startswith('110'), 'site'] = 'UKB'
data.loc[data['site_id'].astype(str).str.startswith('70'), 'site'] = 'IMAGEN'
data.loc[data['site_id'].astype(str).str.startswith('40'), 'site'] = 'OASIS3'

#%% OASIS
batch_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_demented_adults/batch_2/'
with open(os.path.join(batch_dir, "velocity_objects_old.pkl"), "rb") as f:
    old = pickle.load(f)

A3 = new["A_sparse"].toarray()

A_loaded = loaded["A"]
Phi_loaded = loaded["Phi"]
pred_r_loaded = loaded["pred_r"]



# def calculate_z_gain(Z2: float, Z1: float, r: float) -> float:
#     """
#     Compute the z-transformed gain between two repeated measurements (Z1 and Z2),
#     adjusted by the correlation coefficient r.

#     Parameters:
#     - Z2 (float): Measurement at second time point, in z-space
#     - Z1 (float): Measurement at first time point, in z-space
#     - r (float): Correlation coefficient between the two conditions (range -1 to 1)

#     Returns:
#     - z_gain (float): The standardized difference between Z2 and Z1,
#                       adjusted for dependency (correlation) between them.
#     """
#     sigma = np.sqrt(1 - r**2)  # Standard error adjusted for correlation
#     z_gain = (Z2 - r* Z1) / sigma  # Standardized gain
#     return z_gain
    
#%%
idp_cols = DK_idp_cols()
for i, region  in enumerate(idp_cols):
    with open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_DK_demented_adults/batch_{i+1}/velocity_objects_new.pkl", "rb") as f:
        objects = pickle.load(f)
        sparse_banded = objects["sparse_banded"]
        dense = sparse_banded.toarray()
        dense[dense == 0] = np.nan

        # Plot as heatmap
        plt.imshow(dense, cmap='viridis', interpolation='none', vmin=0.7, vmax=1.0)
        plt.colorbar(label='Value')
        plt.title(region)
        plt.xlabel("age")
        plt.ylabel("age")
        plt.show()
#%%
for i, region in enumerate(measures):
    if i == 7:  # break after 5 iterations (index 0 to 4)
        break
    # Load first matrix
    #with open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_DK_demented_adults/batch_{i+1}/velocity_objects_new.pkl", "rb") as f:
    with open(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_DK_retrain/batch_{i+1}/velocity_objects_new.pkl", "rb") as f:
       
        objects = pickle.load(f)
        A1 = objects["A_sparse"].toarray()
        A1[A1 == 0] = np.nan
        A2 = objects["A_sparse_predict"].toarray()
        A2[A2 == 0] = np.nan

    # Plot both side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im1 = axes[0].imshow(A1, cmap='viridis', interpolation='none')#, vmin=0.7, vmax=1.0)
    axes[0].set_title(f"{region} observed")
    axes[0].set_xlabel("age")
    axes[0].set_ylabel("age")

    im2 = axes[1].imshow(A2, cmap='viridis', interpolation='none') #, vmin=0.7, vmax=1.0)
    axes[1].set_title(f"{region} (predicted)")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("age")

    # Shared colorbar to the right of both axes
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, location='right')
    cbar.set_label('Value')

    plt.suptitle(f"Comparison: {region}", y=1.02)
    plt.show()
#%%
sparse_banded = objects["sparse_banded"]
dense = sparse_banded.toarray()
dense[dense == 0] = np.nan

# Plot as heatmap
plt.imshow(dense, cmap='viridis', interpolation='none', vmin=0.7, vmax=1.0)
plt.colorbar(label='Value')
plt.title("Sparse Banded Matrix Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()

#%%

# df = pd.read_csv("your_data.csv")

# Example simulated data
df = pd.DataFrame({
    'height_t1': [174.364088, 162.940028, 171.679470, 175.956054, 161.025318, 160.738237, 161.476885, 162.428823, 171.883497, 154.893211, 160.419067, 165.275191, 154.460142, 158.572423, 158.366244, 175.175718, 165.053873, 156.925864, 156.854309, 167.202583],
    'height_t2': [172.921008, 164.302830, 171.014292, 179.025884, 156.744378, 154.726082, 157.837718, 160.505914, 175.768305, 151.653964, 160.408582, 162.025262, 160.710159, 156.525548, 160.019944, 170.814393, 161.632338, 160.779353, 157.166729, 166.765981],
    'height_t3': [167.571270, 162.689062, 165.945211, 174.249265, 166.721095, 155.451536, 152.620201, 155.191401, 168.659527, 160.703703, 154.591930, 154.589859, 163.894405, 156.288906, 159.121447, 171.373128, 163.413647, 157.770280, 158.356883, 167.709208],
    'height_t4': [168.649110, 165.760032, 180.344250, 177.120677, 157.807505, 162.048343, 151.879707, 157.865620, 162.680938, 150.662920, 164.947111, 154.726354, 160.434176, 156.749259, 161.837737, 167.932684, 160.346655, 163.292582, 158.337591, 166.173047],
    'height_t5': [160.059546, 160.672840, 169.077884, 174.614358, 161.529466, 165.504032, 161.251826, 163.754026, 157.812203, 148.019415, 168.438666, 157.256444, 168.329798, 160.153831, 162.054782, 165.024018, 157.236997, 165.228844, 151.331104, 167.325914],
    'height_t6': [176.295397, 156.342257, 182.699240, 169.015490, 167.353323, 160.081256, 159.840082, 165.356255, 169.038084, 160.636592, 167.011377, 164.265361, 164.606529, 157.302650, 158.689936, 163.455385, 157.598479, 157.794636, 149.967725, 161.645963],
    'height_t7': [172.014281, 175.381642, 157.009536, 181.892003, 164.997411, 161.003355, 165.341960, 169.960962, 160.251207, 152.539145, 165.667458, 155.506241, 161.173615, 155.515678, 165.094426, 162.484143, 166.154530, 157.788903, 157.168574, 157.556735],
    'height_t8': [160.220325, 171.488203, 183.062841, 172.023475, 172.292228, 157.928663, 166.882181, 172.075999, 167.674227, 160.077211, 163.351284, 159.878400, 157.758973, 160.067734, 166.607503, 165.552420, 159.840104, 160.345662, 152.603075, 164.239154],
})
# Compute correlation matrix between timepoints
corr_matrix = df.corr()

# Optional: Visualize with a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation between Height Timepoints")
plt.show()

#%%

# Assume df contains height_t1 to height_t8
lags = []
correlations = []

for lag in range(1, 8):
    corr = df[f'height_t{1}'].corr(df[f'height_t{1+lag}'])
    lags.append(lag)
    correlations.append(corr)

plt.plot(lags, correlations, marker='o')
plt.xlabel('Lag (timepoint difference)')
plt.ylabel('Correlation with t1')
plt.title('Correlation vs Time Lag')
plt.grid(True)
plt.show()

A = objects['A']

bandwidth = 10

# Collect diagonals and their offsets
diagonals = []
offsets = []

for k in range(-bandwidth, bandwidth + 1):
    diag = np.diag(A, k=k)
    diagonals.append(diag)
    offsets.append(k)

# Create sparse banded matrix
A_sparse_banded = diags(diagonals, offsets, shape=A.shape, format='csr')

plt.spy(A_sparse_banded, markersize=5)
plt.title("Sparsity pattern of banded matrix")
plt.show()

k = 2  # offset
diag_k = A_sparse_banded.diagonal(k)  # 2nd upper diagonal
print(diag_k)

import pickle
import pandas as pd
test = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_demented_adults_ADNI/batch_10/cross_sectional.pkl")
