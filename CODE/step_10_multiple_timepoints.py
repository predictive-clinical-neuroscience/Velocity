#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:48:31 2025

@author: johbay
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import SC_idp_cols
import pickle
from collections.abc import Sequence
from typing import Any

#%%
def sweep_operator(A, k):
    """
    Sweep operator matching statistical definition (R fastmatrix::sweep.operator).
    Preserves signs of beta weights like in R.
    """
    A = A.copy().astype(float)
    if isinstance(k, (int, np.integer)):
        k = [k]
    for idx in k:
        Akk = A[idx, idx]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i != idx and j != idx:
                    A[i, j] -= A[i, idx] * A[idx, j] / Akk
        for i in range(A.shape[0]):
            if i != idx:
                A[i, idx] /= Akk
        for j in range(A.shape[1]):
            if j != idx:
                A[idx, j] /= Akk
        A[idx, idx] = 1.0 / Akk  # No negation here
    return A

def betas_from_corr(R_full, predictors, outcome, return_r2=True):
    # extract predictors + outcome into a dense block
    idx = list(predictors) + [outcome]
    idx = [int(x) for x in idx]
    R_block = R_full[np.ix_(idx, idx)]
    np.fill_diagonal(R_block, 1)
    R_block = R_block + R_block.T - np.diag(np.diag(R_block))
    # sweep predictors
    R_swept = sweep_operator(R_block, k=list(range(len(predictors))))

    betas = R_swept[:len(predictors), -1]
    
    if return_r2:
        r2 = 1.0 - R_swept[-1, -1]  # since corr(y,y)=1
        return betas, r2
    
    # betas are in predictor–outcome column after sweep
    return betas, r2

# Example: full sparse/banded corr matrix
np.random.seed(42)
R_full = np.eye(100)
# (fill in banded correlations for illustration)
for i in range(100):
    for j in range(i-2, i+3):
        if 0 <= j < 100 and i != j:
            R_full[i, j] = R_full[j, i] = 0.3
            

cor = np.array([[1.0, 0.856, 0.691],
         [0.856, 1.0, 0.745],
         [0.691, 0.745, 1.0]])

betas = sweep_operator(cor, k=[0,1])

# pick predictors [0, 1, 2], outcome=10
betas = betas_from_corr(R_full, [0, 1, 2], 10)
print(betas)

#%%
np.random.seed(2)
X = np.random.randn(100, 2)
y = 0.4*X[:,0] - 0.3*X[:,1] + np.random.randn(100)

# Correlation matrix
data = np.column_stack([X, y])
R = np.corrcoef(data, rowvar=False)

# Sweep predictors (0,1,2)
R_swept = sweep_operator(R, [0,1])
betas = R_swept[:2, -1]
print("Betas:", betas)
#%%
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X, y)
print(lr.coef_)

#%%
from scipy.sparse import diags
np.random.seed(42)
n = 10       # number of variables
bandwidth = 2  # how many off-diagonals to keep non-zero

#%%
# Values for diagonals: main diagonal = 1 (correlation with self)
diagonals = [np.ones(n)]
# Off-diagonals: decay with distance from main diagonal
for k in range(1, bandwidth + 1):
    val = 0.5 / k   # example correlation decay
    diagonals.append(np.full(n - k, val))
    diagonals.append(np.full(n - k, val))

# Create sparse banded matrix
offsets = [0]
for k in range(1, bandwidth + 1):
    offsets.extend([k, -k])

R_sparse = diags(diagonals, offsets, shape=(n, n), format="csr")

# Convert to dense just to inspect
R_dense = R_sparse.toarray()

print("Sparse matrix:\n", R_dense)

#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
data_in_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed'
idx=1
atlas ="SC"
base_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity"
z_scores = pd.read_pickle(f"{base_path}/SHASHb_1_estimate_scaled_fixed_{atlas}_demented_adults_ADNI/batch_{idx}/Z_estimate.pkl")
test_raw = pd.read_pickle(f'{base_path}/DATA_pre_processed/test_{atlas}_demented_adults_ADNI.pkl')

test_raw.reset_index(drop=True, inplace=True)

data_all = pd.concat([
    z_scores, 
    test_raw[["ID_subject", "ID_visit", "age", "sex", "which_dataset", "site_id"]]
], axis=1)

data_clinical = data_all[data_all['which_dataset'].isin(['4','5'])]

ADNI_clinical = pd.read_pickle('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/ADNI_SC_clinical_goodsites.pkl')
OASIS3_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_SC_demented.pkl")).rename(columns={"diagnosis": "DIAGNOSIS"})

diagnosis = pd.concat(
    [ADNI_clinical[["DIAGNOSIS","ID_subject","ID_visit"]], OASIS3_full_long_demented[["DIAGNOSIS", "ID_subject","ID_visit"]]],
    axis=0,
    ignore_index=True
)

data_clinical = pd.merge(
    data_clinical,
    diagnosis,
    on=['ID_subject', 'ID_visit'],
    how='left'
)

#%%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Count visits per subject and filter for > 3 visits
subjects_with_3plus = (
    data_clinical.groupby("ID_subject")["ID_visit"]
      .count()
      .loc[lambda x: x == 12]
      .index
)

df_filtered = data_clinical[data_clinical["ID_subject"].isin(subjects_with_3plus)]
df_filtered = df_filtered.sort_values(["ID_subject", "ID_visit"])

# Unique diagnoses (excluding NaN)
diagnosis_codes = df_filtered["DIAGNOSIS"].dropna().unique()

# Assign colors
color_map = {diag: color for diag, color in zip(diagnosis_codes, plt.cm.tab10.colors)}
color_map[np.nan] = "lightgrey"  # explicitly handle NaN

# Function to map diagnosis to colors (NaN -> grey)
def diagnosis_to_color(diag):
    if pd.isna(diag):
        return "lightgrey"
    return color_map.get(diag, "lightgrey")

plt.figure(figsize=(8, 5))

for subject_id, sub_df in df_filtered.groupby("ID_subject"):
    # plot the trajectory line
    plt.plot(sub_df["age"], sub_df[0], color="gray", alpha=0.5)
    
    # overlay points with diagnosis color
    plt.scatter(
        sub_df["age"],
        sub_df[0],
        c=sub_df["DIAGNOSIS"].apply(diagnosis_to_color),
        edgecolor="black"
    )
    
    # get last point coordinates for label
    last_age = sub_df["age"].iloc[-1]
    last_score = sub_df[0].iloc[-1]
    
    # add subject ID text slightly offset so it doesn't overlap the marker
    plt.text(
        last_age + 0.5,  # shift right
        last_score, 
        str(subject_id),
        fontsize=8,
        color="black",
        alpha=0.7
    )

# Legend
for diag in diagnosis_codes:
    plt.scatter([], [], color=color_map[diag], edgecolor="black", label=str(diag))
plt.scatter([], [], color="lightgrey", edgecolor="black", label="Unknown")

plt.xlabel("Age")
plt.ylabel("Score")
plt.title("Trajectories for Subjects with >3 Visits (Colored by Diagnosis)")
plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc="best")
plt.tight_layout()
plt.show()


#%%
def compute_Zg_from_subject(
    data_subject: pd.DataFrame,
    velocity_path: str,
    n_predictors: int = 1,
    n_outcome: int = 2,                 # 1-based row index after sorting by age
    measure_col: int | str = 0,         # column index or name
    zg_list: Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Compute Zg at a chosen outcome row using the previous n_predictors rows,
    OR invert the relationship: given zg_list, compute implied outcome value(s).

    Parameters
    ----------
    data_subject : pd.DataFrame
        Must contain 'age' and the target column (by index or name).
    velocity_path : str
        Path to a pickle containing a dict with 'A_sparse_predict' (or 'A_sparse').
    n_predictors : int, default=1
        Number of rows immediately before the outcome used as predictors.
    n_outcome : int, default=2
        1-based index of the outcome row in the sorted-by-age data.
    measure_col : int | str, default=0
        Column index or name for the target measure.
    zg_list : Sequence[float] | None, default=None
        If provided, compute outcome value(s) implied by these zg values.

    Returns
    -------
    dict[str, Any]
        If zg_list is None:
            {'zg', 'outcome_val', 'sigma', 'predictor_sum', 'betas', 'r2'}
        Else:
            {'zg_list', 'outcome_vals', 'sigma', 'predictor_sum', 'betas', 'r2'}
    """
    if n_predictors < 1:
        raise ValueError("n_predictors must be >= 1")

    # Convert to 0-based index internally
    pos_outcome = n_outcome - 1

    # Sort by age and validate indices
    data_subject = data_subject.sort_values("age").reset_index(drop=True)
    n_rows = len(data_subject)
    if pos_outcome >= n_rows:
        raise ValueError(
            f"n_outcome={n_outcome} -> index {pos_outcome}, "
            f"but data has only {n_rows} rows after sorting."
        )
    if pos_outcome - n_predictors < 0:
        raise ValueError(
            f"Need at least {n_predictors} predictor row(s) immediately before row {n_outcome}."
        )

    # Load velocity and get dense A
    velocity = pd.read_pickle(velocity_path)
    A_sp = velocity.get("A_sparse_predict", velocity.get("A_sparse"))
    if A_sp is None:
        raise KeyError("velocity pickle must contain 'A_sparse_predict' or 'A_sparse'")

    if hasattr(A_sp, "toarray"):
        A_dense = A_sp.toarray()
    elif hasattr(A_sp, "todense"):
        A_dense = np.asarray(A_sp.todense())
    else:
        A_dense = np.asarray(A_sp)

    # Ages
    age_pred = data_subject["age"].iloc[pos_outcome - n_predictors : pos_outcome].to_numpy(float)
    age_out = float(data_subject["age"].iloc[pos_outcome])

    # Target series (by index or name)
    y_series = data_subject.iloc[:, measure_col] if isinstance(measure_col, int) else data_subject[measure_col]

    # betas_from_corr should return (betas, r2)
    betas, r2 = betas_from_corr(A_dense, age_pred, age_out)  # noqa: F821 (assumed available)
    betas = np.asarray(betas, dtype=float).reshape(-1)
    if betas.shape[0] != n_predictors:
        raise ValueError(
            f"betas length ({betas.shape[0]}) != n_predictors ({n_predictors})."
        )

    # sigma = sqrt(1 - R^2), clamped
    sigma = float(np.sqrt(max(1e-12, 1.0 - float(r2))))

    # Predictive linear combination from observed predictors
    x = y_series.iloc[pos_outcome - n_predictors : pos_outcome].to_numpy(float).reshape(-1)
    predictor_sum = float(np.dot(x, betas))

    if zg_list is None:
        outcome_val = float(y_series.iloc[pos_outcome])
        zg = (outcome_val - predictor_sum) / sigma
        return {
            "zg": float(zg),
            "outcome_val": outcome_val,
            "sigma": sigma,
            "predictor_sum": predictor_sum,
            "betas": betas,
            "r2": float(r2),
        }
    else:
        outcome_vals = [(float(zg) * sigma) + predictor_sum for zg in zg_list]
        return {
            "zg_list": [float(z) for z in zg_list],
            "outcome_vals": outcome_vals,
            "sigma": sigma,
            "predictor_sum": predictor_sum,
            "betas": betas,
            "r2": float(r2),
        }

#%%
def average_two_rows(df, match_col, value_col, keep='one'):
    """
    Average two rows in `value_col` when `match_col` is the same.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    match_col : str
        Column name to match on.
    value_col : str
        Column whose values should be averaged.
    keep : {'one', 'both'}, default='one'
        If 'one', return one row per group (averaged).
        If 'both', return original rows with a new column for the average.

    Returns
    -------
    pd.DataFrame
        DataFrame with averaged values.
    """
    if keep == 'one':
        averaged = df.groupby(match_col, as_index=False)[value_col].mean()
        return averaged
    elif keep == 'both':
        avg_map = df.groupby(match_col)[value_col].transform('mean')
        df[f'{value_col}_avg'] = avg_map
        return df
    else:
        raise ValueError("keep must be 'one' or 'both'")
        
        

def plot_subjects(data, **kwargs,):
    save = kwargs.get("save", False)
    last = kwargs.get("last", -1)
    second_last = kwargs.get("second_last", -2)
    subject = kwargs.get('subject', 'default')
    n_back = kwargs.get("n_back", 1)
    diagnosis_codes = data["DIAGNOSIS"].dropna().unique()
    
    # Assign colors
    color_map = {diag: color for diag, color in zip(diagnosis_codes, plt.cm.tab10.colors)}
    color_map[np.nan] = "lightgrey"  # explicitly handle NaN

    # Function to map diagnosis to colors (NaN -> grey)
    def diagnosis_to_color(diag):
        if pd.isna(diag):
            return "lightgrey"
        return color_map.get(diag, "lightgrey")

    #plt.figure(figsize=(8, 5))

    for subject_id, sub_df in data.groupby("ID_subject"):
        # plot the trajectory line
        plt.plot(sub_df["age"], sub_df[0], color="gray", alpha=0.5)
        
        # overlay points with diagnosis color
        plt.scatter(
            sub_df["age"],
            sub_df[0],
            c=sub_df["DIAGNOSIS"].apply(diagnosis_to_color),
            edgecolor="black"
        )
        
        # get last point coordinates for label
        last_age = sub_df["age"].iloc[last]
        last_score = sub_df[0].iloc[last]
        
        
        second_last_age = sub_df["age"].iloc[second_last]
        second_last_score = sub_df[0].iloc[second_last]
        # add subject ID text slightly offset so it doesn't overlap the marker
        plt.text(
            last_age + 0.5,  # shift right
            last_score, 
            str(subject_id),
            fontsize=8,
            color="black",
            alpha=0.7
        )

    # add horizontal lines at z = -3..3
    for z in range(-1, 1):
         plt.axhline(y=z, color="lightgrey", linestyle="--", linewidth=0.8, zorder=0)

    # Legend
    for diag in diagnosis_codes:
        plt.scatter([], [], color=color_map[diag], edgecolor="black", label=str(diag))
        plt.scatter([], [], color="lightgrey", edgecolor="black", label="Unknown")
    
    mint = "#98FF98"
    outcome_vals = kwargs.get("outcome_vals")
    vals = outcome_vals
    plt.plot([second_last_age, last_age], [second_last_score, vals[2]], color=mint, lw=1)
    plt.plot([second_last_age, last_age], [second_last_score, vals[0]], color=mint, lw=1)
    
    
    df = pd.DataFrame({
    "x": [second_last_age, last_age, second_last_age, last_age],
    "y": [second_last_score, vals[2], second_last_score, vals[0]]
    })

    if save:
        # save as pickle
        df.to_pickle(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_data_{subject}_{n_back}.pkl")
        #data.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/lines_df2.pkl")
    
  

    # Fill the triangle between them
    plt.fill(
        [second_last_age, last_age, last_age],
        [second_last_score, vals[2], vals[0]],
        color=mint, alpha=0.3
    )


    plt.xlabel('Age',fontsize=15)
    #plt.xlim(82,92)
    #plt.ylabel('z_score',fontsize=12)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.grid(linestyle=":", linewidth=1, alpha=0.7)
    #fig.axes[0].yaxis.offsetText.set_fontsize(14)
    #plt.ylabel("Score (z)")
    #plt.title("Trajectories for selected subjects from the ADNI dataset (z-space)")
    #plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc="best")
    plt.tight_layout()
    if save:
        plt.savefig(
            f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/trajectories_ADNI_{subject}_{n_back}.png", 
            dpi=300, bbox_inches="tight"
            )
    plt.show()
    
    

#%%
velocity_path = f"{base_path}/SHASHb_1_estimate_scaled_fixed_{atlas}_retrain_ADNI/batch_{idx}/velocity_objects_new.pkl"



#%%
data_subject = data_clinical[data_clinical["ID_subject"] == "099_S_0051"]
df1 = average_two_rows(data_subject, "age", 0, keep="both")
df1.drop_duplicates(subset=['0_avg'], inplace=True)

data_subject = data_clinical[data_clinical["ID_subject"] == "013_S_1186"]
df2 = average_two_rows(data_subject, "age", 0, keep="both")
df2.drop_duplicates(subset=['0_avg'], inplace=True)

# data_subject = data_clinical[data_clinical["ID_subject"] == "023_S_0217"]
# df3 = average_two_rows(data_subject, "age", 0, keep="both")
# df3.drop_duplicates(subset=['0_avg'], inplace=True)

data_subject = data_clinical[data_clinical["ID_subject"] == "018_S_0142"]
df4 = average_two_rows(data_subject, "age", 0, keep="both")
df4.drop_duplicates(subset=['0_avg'], inplace=True)

data_subject = data_clinical[data_clinical["ID_subject"] == "002_S_1070"]
df5 = average_two_rows(data_subject, "age", 0, keep="both")
df5.drop_duplicates(subset=['0_avg'], inplace=True)

#data = pd.concat([df1,df2, df3], axis=0)
data = pd.concat([df1, df2, df4, df5], axis=0)

data = pd.merge(
    data,
    test_raw[['ID_subject','site_id2', 'ID_visit']],
    on=['ID_subject', 'ID_visit'],
    how='left'
)


# save as pickle
data.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/data_4_subjects.pkl")


df1_099_S_0051 = pd.merge(
    df1,
    test_raw[['ID_subject','site_id2', 'ID_visit']],
    on=['ID_subject', 'ID_visit'],
    how='left'
)

df1_099_S_0051.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df1_099_S_0051.pkl")


df2_013_S_1186 = pd.merge(
    df2,
    test_raw[['ID_subject','site_id2', 'ID_visit']],
    on=['ID_subject', 'ID_visit'],
    how='left'
)

df2_013_S_1186.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df2_013_S_1186.pkl")

df4_018_S_0142 = pd.merge(
    df4,
    test_raw[['ID_subject','site_id2', 'ID_visit']],
    on=['ID_subject', 'ID_visit'],
    how='left'
)

df4_018_S_0142.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df4_018_S_0142.pkl")

df5_002_S_1070 = pd.merge(
    df5,
    test_raw[['ID_subject','site_id2', 'ID_visit']],
    on=['ID_subject', 'ID_visit'],
    how='left'
)

df5_002_S_1070.to_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/df5_002_S_1070.pkl")



#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5), dpi=300)  # high resolution when creating figure

for subject_id, sub_df in data.groupby("ID_subject"):
    # plot the trajectory line in grey
    plt.plot(sub_df["age"], sub_df[0], color="gray", alpha=0.5)

    # overlay points with diagnosis color
    plt.scatter(
        sub_df["age"],
        sub_df[0],
        c=sub_df["DIAGNOSIS"].apply(diagnosis_to_color),
        edgecolor="black",
        s=40,            # consistent marker size
        zorder=5         # points stay on top of lines
    )

    # label subject at last point
    last_age = sub_df["age"].iloc[-1]
    last_score = sub_df[0].iloc[-1]
    plt.text(
        last_age + 0.5,  # shift right
        last_score,
        str(subject_id),
        fontsize=10,
        color="black",
        alpha=0.7
    )

# reference lines
plt.hlines(y=-1, xmin=65, xmax=92, colors="black", linestyles="--", linewidth=1)
plt.hlines(y=0,  xmin=65, xmax=92, colors="black", linestyles="-",  linewidth=1.5)
plt.hlines(y=1,  xmin=65, xmax=92, colors="black", linestyles="--", linewidth=1)

# Legend (diagnosis categories)
for diag in diagnosis_codes:
    plt.scatter([], [], color=color_map[diag], edgecolor="black", label=str(diag))

# Axis labels and ticks
plt.xlim(65,92)
plt.title("z-score  Left-Lateral-Ventricle", fontsize = 16)
plt.xlabel("Age", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)

# Gridlines
plt.grid(linestyle=":", linewidth=1, alpha=0.7)

# Legend
plt.legend(title="Diagnosis", loc="upper right", framealpha=0.8)

# Layout & save
plt.tight_layout()
#plt.savefig("trajectories_highres_all4.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
outcome1_1 = compute_Zg_from_subject(df1, velocity_path, n_predictors=1, n_outcome=5, 
                                  zg_list=[-2.0, 0.0, 2.0] )
outcome1_4 = compute_Zg_from_subject(df1, velocity_path, n_predictors=4, n_outcome=5, 
                                  zg_list=[-2.0, 0.0, 2.0] )

outcome1_4_vals = outcome1_4['outcome_vals']
plot_subjects(df1, outcome_vals=outcome1_4_vals, subject="099_S_0051", n_back=4, last =-3, second_last=-4, save=True)

outcome1_1_vals = outcome1_1['outcome_vals']
plot_subjects(df1, outcome_vals=outcome1_1_vals, subject="099_S_0051", n_back=1, last =-3, second_last=-4, save=True)

#%%

outcome2_6 = compute_Zg_from_subject(df2, velocity_path, n_predictors=6, n_outcome=7, 
                                  zg_list=[-2.0, 0.0, 2.0] )
outcome2_1 = compute_Zg_from_subject(df2, velocity_path, n_predictors=1, n_outcome=7, 
                                  zg_list=[-2.0, 0.0, 2.0] )

outcome2_1_vals = outcome2_1['outcome_vals']
plot_subjects(df2, outcome_vals=outcome2_1_vals, save = True, last = -1, second_last =- 2, subject="013_S_1186", n_back= 1)

outcome2_6_vals = outcome2_6['outcome_vals']
plot_subjects(df2, outcome_vals=outcome2_6_vals, save = True, last = -1, second_last =- 2, subject="013_S_1186", n_back= 6)


#%%

outcome4_6 = compute_Zg_from_subject(df4, velocity_path, n_predictors=6, n_outcome=7, 
                                  zg_list=[-2.0, 0.0, 2.0])

outcome4_1 = compute_Zg_from_subject(df4, velocity_path, n_predictors=1, n_outcome=7, 
                                  zg_list=[-2.0, 0.0, 2.0] )

outcome4_6_vals = outcome4_6['outcome_vals']
plot_subjects(df4, outcome_vals = outcome4_6_vals,save=True, subject= "018_S_0142", n_back=6)

outcome4_1_vals = outcome4_1['outcome_vals']
plot_subjects(df4, outcome_vals = outcome4_1_vals,save=True, subject= "018_S_0142", n_back=1)

outcome5_1 = compute_Zg_from_subject(df5, velocity_path, n_predictors=1, n_outcome=3, 
                                  zg_list=[-2.0, 0.0, 2.0])

outcome5_2 = compute_Zg_from_subject(df5, velocity_path, n_predictors=2, n_outcome=3,
                                  zg_list=[-2.0, 0.0, 2.0])

outcome5_1_vals = outcome5_1['outcome_vals']
plot_subjects(df5, outcome_vals = outcome5_1_vals,save =True, last = -2, second_last =- 3, subject="002_S_1070", n_back= 1)

outcome5_2_vals = outcome5_2['outcome_vals']
plot_subjects(df5, outcome_vals = outcome5_2_vals, save =True, last = -2, second_last =- 3, subject="002_S_1070", n_back= 2)
#%%
#%%
idp_cols = SC_idp_cols()

X_train = data[["age","sex"]].to_numpy(dtype=float)
Z_train = data["site_id2"].to_numpy(dtype=float)
#Y_train= train[measure_name].to_numpy()
Y_train= data[0].to_numpy(dtype=float)



#%%
base_dir = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity"
os.chdir(base_dir)

with open('DATA_NM/X_train_SC_plot_subjects.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train), file)  
with open('DATA_NM/Y_train_SC_plot_subjects.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train), file)
with open('DATA_NM/trbefile_SC_plot_subject.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Z_train), file)

