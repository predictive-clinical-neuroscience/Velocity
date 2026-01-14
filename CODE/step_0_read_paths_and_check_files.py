#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 09:47:09 2025

@author: johbay
"""

import pandas as pd
import numpy as np


train_old =pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC/old/train_retrain_full_models_SC_adults_ADNI.pkl")
train_new =pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/SC/train_retrain_full_models_SC_adults_ADNI.pkl")

Z_thalamus = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_SC_SHASH_train_retrain_Thalamus/results/Z_test.csv")

#%%

test_resp = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_retrain_ADNI/batch_1/testresp_batch_1.pkl")

Y_train_retrain = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_retrain_SC_adults_ADNI.pkl")


#%%
train_retrain = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed/train_retrain_full_models_SC_adults_ADNI.pkl")
train_retrain_new = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_new_version/SC/train_retrain_full_models_SC_adults_ADNI.pkl")


old_z = pd.read_pickle("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/SHASHb_1_estimate_scaled_fixed_SC_retrain_ADNI/batch_1/Z_estimate.pkl")
new_z = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_SC_all_regions2_full_datset/results/Z_test.csv")

new_z_sorted = new_z.sort_values(by="observations")
new_z_sorted = new_z.sort_values(by="observations").reset_index(drop=True)

new_z_Left_Lateral_Ventricle = new_z_sorted["Left-Lateral-Ventricle"].to_frame()
#%%
new_z_Left_Lateral_Ventricle.iloc[:, 0].corr(old_z.iloc[:, 0])

train_retrain["site_id"].corr(
    train_retrain_new["site_id"]
)

train_retrain["site_id2"].corr(
    train_retrain_new["site_id2"]
)

train_retrain["sex"].corr(
    train_retrain_new["sex"]
)


train_retrain["Left-Lateral-Ventricle"].corr(
    train_retrain_new["Left-Lateral-Ventricle"]
)
#%%
new_z_Left_Lateral_Ventricle["Left-Lateral-Ventricle"].cor(old_z["0"])

train_retrain[:,"Left-Lateral-Ventricle"].cor(train_retrain_new[:,"Left-Lateral-Ventricle"])
#%%
np.corrcoef(new_z_Left_Lateral_Ventricle, old_z)


#%%
from pathlib import Path
from PIL import Image

# ---- settings ----
png_folder = Path("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/")
output_pdf = "summary_limited_DES.pdf"

cols = 2
rows = 5
per_page = cols * rows

# A4 at ~150 DPI
page_w, page_h = 1240, 1754
bg_color = "white"

# Cell sizes
cell_w = page_w // cols
cell_h = page_h // rows

# ---- collect images ----
png_files = sorted(
    list(png_folder.glob("1_lh*new_version.png")) +
    list(png_folder.glob("2_lh*new_version.png")) +
    list(png_folder.glob("4_lh*new_version.png")) +
    list(png_folder.glob("1_rh*new_version.png")) +
    list(png_folder.glob("2_rh*new_version.png")) +
    list(png_folder.glob("4_rh*new_version.png"))
)

if not png_files:
    raise RuntimeError(f"No PNGs found in {png_folder}")

pages = []

# ---- build pages ----
for start in range(0, len(png_files), per_page):
    batch = png_files[start:start + per_page]

    # blank A4 page
    page = Image.new("RGB", (page_w, page_h), bg_color)

    for idx, img_path in enumerate(batch):
        img = Image.open(img_path).convert("RGB")

        # resize into the cell
        img.thumbnail((cell_w, cell_h), Image.LANCZOS)

        col = idx % cols
        row = idx // cols

        x = col * cell_w + (cell_w - img.width) // 2
        y = row * cell_h + (cell_h - img.height) // 2

        page.paste(img, (x, y))

    pages.append(page)

# ---- save PDF ----
first, *rest = pages
first.save(output_pdf, save_all=True, append_images=rest)

print(f"Saved {len(pages)} pages to {output_pdf} (A4 size)")