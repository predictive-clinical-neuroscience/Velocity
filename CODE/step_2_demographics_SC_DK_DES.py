#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:51:52 2025

@author: johbay
"""
#%% get the stats for the paper
import pcntoolkit as ptk
from utils import ldpkl
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
#%%
data_in_dir = '/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/'

#%%
train_DK = ldpkl(os.path.join(data_in_dir, "DK/train_DK_demented_adults_ADNI.pkl"))
test_DK = ldpkl(os.path.join(data_in_dir, "DK/test_DK_demented_adults_ADNI.pkl"))

train_DK.shape[0] + test_DK.shape[0] #69820
#%%
train_SC = ldpkl(os.path.join(data_in_dir, "SC/train_SC_demented_adults_ADNI.pkl"))
test_SC = ldpkl(os.path.join(data_in_dir, "SC/test_SC_demented_adults_ADNI.pkl"))

train_SC.shape[0] + test_SC.shape[0] #79280
#%%
train_DES = ldpkl(os.path.join(data_in_dir, "DES/train_DES_demented_adults_ADNI.pkl"))
test_DES = ldpkl(os.path.join(data_in_dir, "DES/test_DES_demented_adults_ADNI.pkl"))

train_DES.shape[0] + test_DES.shape[0] # 74486

#%%
train_DES_retrain = ldpkl(os.path.join(data_in_dir, "DES/train_retrain_full_models_DES_adults_ADNI.pkl"))
train_DK_retrain = ldpkl(os.path.join(data_in_dir, "DK/train_retrain_full_models_DK_adults_ADNI.pkl"))
train_SC_retrain = ldpkl(os.path.join(data_in_dir, "SC/train_retrain_full_models_SC_adults_ADNI.pkl"))

train_DES_retrain.shape[0]
train_DK_retrain.shape[0]
train_SC_retrain.shape[0]

len(train_DES_retrain["ID_subject"].unique())
len(train_DK_retrain["ID_subject"].unique())
len(train_SC_retrain["ID_subject"].unique())
#%% Full sample age range
train_DES_retrain["age"].min()
train_DES_retrain["age"].max()

len(train_DES_retrain[train_DES_retrain["which_dataset"]=="1"]) +len(train_DES_retrain[train_DES_retrain["which_dataset"]=="3"])
len(train_DES_retrain[train_DES_retrain["which_dataset"]=="2"])

long = train_DES_retrain[train_DES_retrain["which_dataset"]=="2"]
len(long["ID_subject"].unique())

train_DES.shape[0]
test_DES.shape[0]

#%% Full sample age range
train_DK_retrain["age"].min()
train_DK_retrain["age"].max()

len(train_DK_retrain[train_DK_retrain["which_dataset"]=="1"]) +len(train_DK_retrain[train_DK_retrain["which_dataset"]=="3"])
len(train_DK_retrain[train_DK_retrain["which_dataset"]=="2"])

long = train_DK_retrain[train_DK_retrain["which_dataset"]=="2"]
len(long["ID_subject"].unique())

test_DK.shape[0]
train_DK.shape[0]

#%% Full sample age range
train_SC_retrain["age"].min()
train_SC_retrain["age"].max()

len(train_SC_retrain[train_SC_retrain["which_dataset"]=="1"]) +len(train_SC_retrain[train_SC_retrain["which_dataset"]=="3"])
len(train_SC_retrain[train_SC_retrain["which_dataset"]=="2"])

long = train_SC_retrain[train_SC_retrain["which_dataset"]=="2"]
len(long["ID_subject"].unique())

test_SC.shape[0]
train_SC.shape[0]

#%% OASIS sample
OASIS2_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "DES/OASIS2_demented_long_DES.pkl"))

OASIS3_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "DES/OASIS3_demented_long_DES.pkl"))

len(OASIS3_full_long_demented["ID_subject"].unique()) # 146 scans
len(OASIS2_full_long_demented["ID_subject"].unique()) # 181 scans

#%%
from step_9_2_ADNI_OASIS_z_gain_new_version import load_adni_data

z_scores_ADNI_SC = load_adni_data(atlas="SC")
z_scores_ADNI_SC["DIAGNOSIS"].value_counts().loc[[2, 3]].sum()

z_scores_ADNI_DK = load_adni_data(atlas="DK")
z_scores_ADNI_DK["DIAGNOSIS"].value_counts().loc[[2, 3]].sum()

z_scores_ADNI_DES = load_adni_data(atlas="DES")
z_scores_ADNI_DES["DIAGNOSIS"].value_counts().loc[[2, 3]].sum()


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker


# --- INPUT ---
# long must have: ID_subject, age, ID_visit
df = long.copy()

# --- ENSURE VISIT IS NUMERIC (REQUIRED) ---
df["ID_visit"] = pd.to_numeric(df["ID_visit"], errors="raise")

# --- SORT SO LINES CONNECT IN TIME ORDER ---
df = df.sort_values(["ID_subject", "age"])

# --- MAP SUBJECTS TO Y POSITIONS ---
subjects = pd.Index(df["ID_subject"].unique())
y_map = {sid: i for i, sid in enumerate(subjects)}
df["y"] = df["ID_subject"].map(y_map)

# --- DISCRETE VISIT COLORS ---
visits = sorted(df["ID_visit"].unique())
n_visits = len(visits)

cmap = mpl.colors.ListedColormap(
    plt.get_cmap("Set2", n_visits).colors
)
norm = mpl.colors.BoundaryNorm(
    boundaries=np.arange(n_visits + 1) - 0.5,
    ncolors=n_visits
)

# map visit → 0..n-1 (required for discrete mapping)
visit_to_idx = {v: i for i, v in enumerate(visits)}
df["visit_idx"] = df["ID_visit"].map(visit_to_idx)

# --- PLOT ---
fig, ax = plt.subplots(figsize=(10, 20))

for sid, g in df.groupby("ID_subject", sort=False):
    # ax.plot(
    #     g["age"].to_numpy(),
    #     g["y"].to_numpy(),
    #     color="black",
    #     linewidth=0.8,
    #     alpha=0.7,
    #     zorder=1
    # )
    ax.scatter(
        g["age"].to_numpy(),
        g["y"].to_numpy(),
        c=g["visit_idx"].to_numpy(),
        cmap=cmap,
        norm=norm,
        s=20,
        zorder=2
    )

# --- DISCRETE COLORBAR ---
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    ticks=range(n_visits)
)
cbar.set_label("Visit")
cbar.set_ticklabels([f"Visit {int(v)}" for v in visits])

# --- CLEAN AXES ---
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
#plt.show()
plt.savefig(fname = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/longitudinal_visits.png", dpi="figure")
#%%
site_to_id = site_dictionary()

for key, values in site_to_id.items():
    if values[1] == 'ABCD':
        site_to_id[key] = (values[1], values[1])
        
for key, values in site_to_id.items():
    if values[1] == 'ADNI':
        site_to_id[key] = (values[1], values[1])
#%%

# df = train_SC_retrain.copy()
# df["site_id"] = df["site_id"].astype(int)
# # Now map the values
# df['site_name'] = df['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
# df['dataset_name'] = df['site_id'].map(lambda x: site_to_id.get(x, (None, None))[1])


#%%

# train_SC['site_name'] = train_SC['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
# test_SC['site_name'] = test_SC['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])


# #%% Ridge plot
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde


# datasets = {
#     "test_SC": test_SC,
#     "train_SC": train_SC
# }
# #%%
# import os
# os.makedirs("results", exist_ok=True)

# for name, df in datasets.items():
#     # ----------- Filter and preprocess ----------
#     df = df[(df["age"] >= 0) & (df["age"] <= 120)]

#     site_counts = df["site_name"].value_counts()
#     top_sites = site_counts[site_counts >= 30].index
#     df = df[df["site_name"].isin(top_sites)]

#     df["site_name"] = df["site_name"].astype(str)
#     mean_ages = df.groupby("site_name")["age"].mean().sort_values()
#     df["site_name"] = pd.Categorical(df["site_name"], categories=mean_ages.index, ordered=True)

#     # ----------- Plot ----------
#     sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
#     pal = sns.cubehelix_palette(len(mean_ages), rot=-.25, light=.7)

#     g = sns.FacetGrid(df, row="site_name", hue="site_name", aspect=15, height=0.5, palette=pal)

#     def kde_normalized(x, color, label):
#         ax = plt.gca()
#         x_grid = np.linspace(x.min(), x.max(), 200)
#         kde = gaussian_kde(x)
#         y = kde(x_grid)
#         y = y / y.max()
#         ax.fill_between(x_grid, y, color=color, alpha=0.9)
#         ax.plot(x_grid, y, color="white", lw=0.5)

#     g.map(kde_normalized, "age")

#     def label(x, color, label):
#         ax = plt.gca()
#         ax.text(-0.02, 0.1, f"Site {label}", fontweight="normal", fontsize=10, color="black",
#                 ha="right", va="center", transform=ax.transAxes, clip_on=False)

#     g.map(label, "age")

#     g.figure.subplots_adjust(hspace=-0.25)
#     g.set_titles("")
#     g.set(yticks=[], ylabel="")
#     g.despine(bottom=True, left=True)
#     plt.xlabel("Age", fontsize=12, fontweight="bold")

#     # ----------- Save for each dataset ----------
#     #output_path = f"results/normalized_ridge_plot_{name}.png"
#     plt.tight_layout()
#     #plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.savefig(f'/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/age_ridge_{name}.png')
#     plt.show()

#%%
dataset_SC = pd.concat([train_SC, test_SC])
dataset_SC['site_name'] = dataset_SC['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])

dataset_DK = pd.concat([train_DK, test_DK])
dataset_DK['site_name'] = dataset_DK['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])

dataset_DES = pd.concat([train_DES, test_DES])
dataset_DES['site_name'] = dataset_DES['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])


#%%
dataset_SC["sex"] = dataset_SC["sex"].replace({0: "F", 1: "M"})

# Group by site_id, sex, and which_dataset
summary = (
    dataset_SC.groupby(["which_dataset", "site_name", "sex"])
      .agg(
          n_subjects=("age", "count"),
          age_min=("age", "min"),
          age_max=("age", "max")
      )
      .reset_index()
)

summary["n_site_total"] = (
    summary
    .groupby(["which_dataset", "site_name"])["n_subjects"]
    .transform("sum")
)

summary["percent"] = (
    100 * summary["n_subjects"] / summary["n_site_total"]
)
# Optional: sort
summary = summary.sort_values(by=["which_dataset", "site_name", "sex"])
summary = summary.dropna()

summary.to_csv("../results/demographics_by_site_SC.csv", index=False)

#%%
dataset_DK["sex"] = dataset_DK["sex"].replace({0: "F", 1: "M"})

# Group by site_id, sex, and which_dataset
summary = (
    dataset_DK.groupby(["which_dataset", "site_name", "sex"])
      .agg(
          n_subjects=("age", "count"),
          age_min=("age", "min"),
          age_max=("age", "max")
      )
      .reset_index()
)

summary["n_site_total"] = (
    summary
    .groupby(["which_dataset", "site_name"])["n_subjects"]
    .transform("sum")
)

summary["percent"] = (
    100 * summary["n_subjects"] / summary["n_site_total"]
)
# Optional: sort
summary = summary.sort_values(by=["which_dataset", "site_name", "sex"])
summary = summary.dropna()

summary.to_csv("../results/demographics_by_site_DK.csv", index=False)

#%%
dataset_DES["sex"] = dataset_DES["sex"].replace({0: "F", 1: "M"})

# Group by site_id, sex, and which_dataset
summary = (
    dataset_DES.groupby(["which_dataset", "site_name", "sex"])
      .agg(
          n_subjects=("age", "count"),
          age_min=("age", "min"),
          age_max=("age", "max")
      )
      .reset_index()
)

summary["n_site_total"] = (
    summary
    .groupby(["which_dataset", "site_name"])["n_subjects"]
    .transform("sum")
)

summary["percent"] = (
    100 * summary["n_subjects"] / summary["n_site_total"]
)
# Optional: sort
summary = summary.sort_values(by=["which_dataset", "site_name", "sex"])
summary = summary.dropna()

summary.to_csv("../results/demographics_by_site_DES.csv", index=False)


#%%
site_to_id = site_dictionary()

# Now map the values
summary['site_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
summary['dataset_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[1])

#%% Sex all

# Load and filter your dataset
df = pd.read_pickle(
    "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/new/train_retrain_full_models_SC_adults_ADNI.pkl"
)

df.groupby("sex").count()/len(df)

df["age"].mean()
df["age"].std()

len(df["ID_subject"].unique())

#%%
df_long= test_SC[test_SC["which_dataset"]=='2']
len(df_long.site_id.unique())

df_long.groupby("sex").count()/len(df_long)

df_long["age"].mean()
df_long["age"].std()

len(df_long["ID_subject"].unique())

#%%
len(df_long.ID_subject.unique())

visit_counts = df_long.groupby("ID_subject").size()
filtered_counts = visit_counts[visit_counts > 1]
plt.style.use('seaborn-v0_8-whitegrid')
# Plot histogram
plt.figure(figsize=(8, 5), facecolor='white')
plt.hist(filtered_counts, bins=range(1, visit_counts.max() + 2), edgecolor='black', color= 'darkred', align='left')
plt.title("Distribution of Maximum Number of Visits per Subject \n (n = 10966 subjects, 26170 scans) " )
plt.xlabel("Number of Visits")
plt.ylabel("Number of Subjects")
plt.xticks(range(2, visit_counts.max() + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/visit_distribution.png', dpi=300)
plt.show()

#%%
df_cross = df[df["which_dataset"]!='2']
len(df_cross.site_id.unique())

df_cross.groupby("sex").count()/len(df_cross)

df_cross["age"].mean()
df_cross["age"].std()

#%%
OASIS2_full_cross_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_cross_SC_demented.pkl"))
OASIS2_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS2_long_SC_demented.pkl"))

OASIS3_full_cross_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_cross_SC_demented.pkl"))
OASIS3_full_long_demented = pd.read_pickle(os.path.join(data_in_dir, "OASIS3_long_SC_demented.pkl"))

len(OASIS2_full_long_demented["ID_subject"].unique())
len(OASIS3_full_long_demented["ID_subject"].unique())

len(OASIS3_full_long_demented["site_id"].unique())

OASIS2_full_long_demented.groupby("sex").count()/len(OASIS2_full_long_demented)
OASIS3_full_long_demented.groupby("sex").count()/len(OASIS3_full_long_demented)

OASIS2_full_long_demented["age"].mean()
OASIS2_full_long_demented["age"].std()

OASIS3_full_long_demented["age"].mean()
OASIS3_full_long_demented["age"].std()

#%%
### adding the actual sites
df = pd.read_pickle(
    "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/train_retrain_full_models_SC_adults.pkl"
)


df["sex"] = df["sex"].replace({0: "F", 1: "M"})

# Group by site_id, sex, and which_dataset
summary = (
    df.groupby(["which_dataset", "site_id", "sex"])
      .agg(
          n_subjects=("age", "count"),
          age_min=("age", "min"),
          age_max=("age", "max")
      )
      .reset_index()
)

# Optional: sort
summary = summary.sort_values(by=["which_dataset", "site_id", "sex"])
summary = summary.dropna()

site_to_id = site_dictionary()

# Now map the values
summary['site_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
summary['dataset_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[1])
#%%
summary.to_csv('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/site_age_sex_table.csv')

#%%
### adding the actual sites


import pandas as pd

# Load your data
df = pd.read_pickle(
    "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC/train_retrain_full_models_SC_adults_ADNI.pkl"
)

# Replace 0/1 coding in sex
df["sex"] = df["sex"].replace({0: "F", 1: "M"})
df["site_id"] = df["site_id"].astype(int)

# Add an is_female column
df["is_female"] = df["sex"].map(lambda x: 1 if x == "F" else 0)


# Group by which_dataset and site_id (NOT is_female yet)
summary = (
    df.groupby(["site_id", "which_dataset"])
      .agg(
          total_subjects=("age", "count"),
          n_female=("is_female", "sum"),  # sum counts all 1s = number of females
          age_min=("age", "min"),
          age_max=("age", "max")
      )
      .reset_index()
)

# Calculate percentage of females
summary["percentage_women"] = (summary["n_female"] / summary["total_subjects"]) * 100

# Optional: round for prettier display
summary["percentage_women"] = summary["percentage_women"].round(1)

# Optional: sort
summary = summary.sort_values(by=["which_dataset", "site_id"]).dropna()

# Done


# Display the result
site_to_id = site_dictionary()

# Now map the values
summary['site_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
summary['dataset_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[1])
#%%
summary.to_csv('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/site_age_table_percentag_women.csv')
#%%
df = pd.read_pickle(
    "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/train_retrain_full_models_SC_adults.pkl"
)

# Replace 0/1 coding in sex
df["sex"] = df["sex"].replace({0: "F", 1: "M"})
df["site_id"] = df["site_id"].astype(int)

# Add an is_female column
df["is_female"] = df["sex"].map(lambda x: 1 if x == "F" else 0)

df['site_name'] = df['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
df['dataset_name'] = df['site_id'].map(lambda x: site_to_id.get(x, (None, None))[1])

summary = (
    df.groupby(["dataset_name", "which_dataset"])
      .agg(
          total_subjects=("age", "count"),
          n_female=("is_female", "sum"),  # sum counts all 1s = number of females
          age_min=("age", "min"),
          age_max=("age", "max")
      )
      .reset_index() 
)

# Calculate percentage of females
summary["percentage_women"] = (summary["n_female"] / summary["total_subjects"]) * 100

# Optional: round for prettier display
summary["percentage_women"] = summary["percentage_women"].round(1)

#%%
summary.to_csv('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/dataset_age_table_percentag_women.csv')
#%%
summary = summary[summary["which_dataset"] == "2"]
summary.to_csv('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/which_dataset_age_table_percentag_women.csv')




#%%

#%%
### adding the actual sites
df = pd.read_pickle(
    "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/train_retrain_full_models_SC_adults.pkl"
)


df["sex"] = df["sex"].replace({0: "F", 1: "M"})
df["site_id"] = df["site_id"].astype(int)

# Group by site_id, sex, and which_dataset
summary = (
    df.groupby(["which_dataset", "site_id", "sex"])
      .agg(
          n_subjects=("age", "count"),
          age_min=("age", "min"),
          age_max=("age", "max")
      )
      .reset_index()
)

# Optional: sort
summary = summary.sort_values(by=["which_dataset", "site_id", "sex"])
summary = summary.dropna()

site_to_id = site_dictionary()

# Now map the values
summary['site_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
summary['dataset_name'] = summary['site_id'].map(lambda x: site_to_id.get(x, (None, None))[1])
#%%
summary.to_csv('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/results/site_age_sex_table.csv')

df["which_dataset"] = df["which_dataset"].astype(int)

data = df[df["which_dataset"]==2]

data["site_id"] = data["site_id"].astype(int)
data['site_name'] = data['site_id'].map(lambda x: site_to_id.get(x, (None, None))[0])
data['dataset_name'] = data['site_id'].map(lambda x: site_to_id.get(x, (None, None))[1])
df["dataset_name"].isna().sum()

OASIS3 = data[data["dataset_name"] =="OASIS3"]
OASIS2 = data[data["dataset_name"] =="OASIS2"]

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df['age'].hist(bins=20, edgecolor='black')

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#%%

import pandas as pd
import plotly.graph_objects as go
import kaleido
#%%
# example edge list
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = ["A1", "A2", "B1", "B2", "C1", "C2"],
      color = "blue"
    ),
    link = dict(
      source = [0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = [2, 3, 3, 4, 4, 5],
      value = [8, 4, 2, 8, 4, 2]
  ))])

fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.show()


# # Load and filter your dataset
# df = pd.read_pickle(
#     "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/train_retrain_full_models_SC_adults.pkl"
# )
# #df = df[2000:20000]
# # Filter out invalid ages and keep only well-populated sites
# df = df[(df["age"] >= 0) & (df["age"] <= 120)]
# #site_counts = df["site_id"].value_counts()
# #top_sites = site_counts[site_counts >= 10].head(100).index
# #df = df[df["site_id"].isin(top_sites)]

# # Convert to string and order sites
# df["site_id"] = df["site_id"].astype(str)
# mean_ages = df.groupby("site_id")["age"].mean().sort_values()
# df["site_id"] = pd.Categorical(df["site_id"], categories=mean_ages.index, ordered=True)

# # Seaborn style
# sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# # Create color palette
# pal = sns.cubehelix_palette(len(mean_ages), rot=-.25, light=.7)

# # Initialize FacetGrid
# g = sns.FacetGrid(df, row="site_id", hue="site_id", aspect=15, height=0.2, palette=pal)

# # KDE plots
# g.map(sns.kdeplot, "age",
#       bw_adjust=1.2, clip_on=False, fill=True, alpha=1, linewidth=0.5)
# g.map(sns.kdeplot, "age",
#       bw_adjust=1.2, clip_on=False, color="w", lw=0.5)

# # Horizontal ref line at y=0
# #g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# # Label function for facet axes
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(-0.02, 0.1, f"Site {label}", fontweight="normal", fontsize=9, color=color,
#             ha="right", va="center", transform=ax.transAxes, clip_on=False)

# g.map(label, "age")

# # Adjust layout
# g.figure.subplots_adjust(hspace=-0.8)
# g.set_titles("")
# g.set(yticks=[], ylabel="")
# g.despine(bottom=True, left=True)
# plt.xlabel("Age", fontsize=12, fontweight="bold")
# plt.show()