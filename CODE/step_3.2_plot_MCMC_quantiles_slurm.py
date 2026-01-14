# coding: utf-8

import os
import gc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages

from pcntoolkit.util.hbr_utils import *
from pcntoolkit.util.utils import scaler
from utils import DES_idp_cols, sc_idp_cols, DK_idp_cols

# Set global plotting options
cols = ['#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']
plt.rc('axes', axisbelow=True)

# Load data
features = DES_idp_cols()
X_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/X_train_retrain_DES_adults.pkl", 'rb')).to_numpy()
Y_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/Y_train_retrain_DES_adults.pkl", 'rb'))
Z_train = pickle.load(open("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_NM/trbefile_retrain_DES_adults.pkl", 'rb')).to_numpy()

# Working directory
os.chdir('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity')


# === Modified plotting function to draw into a given axis ===
def plot_MAP_quantiles(feature="L_bankssts", l='SHASHo', selected_sex='female', atlas='DK', ax=None):
    f_idx = features.index(feature)
    try:
        model_path = f"{l}_estimate_scaled_fixed_{atlas}_retrain/batch_{f_idx+1}/Models/NM_0_0_estimate.pkl"
        model = pd.read_pickle(model_path)

        inscaler = scaler("standardize")
        outscaler = scaler("standardize")
        selected_sex_id = 0 if selected_sex == 'female' else 1
        this_Y_train = Y_train[f_idx].to_numpy()
        this_scaled_X_train = inscaler.fit_transform(X_train)
        this_scaled_Y_train = outscaler.fit_transform(this_Y_train)
        train_sex_idx = np.where(X_train[:, 1] == selected_sex_id)

        model_be = [19]
        sigma_intercept_mu = model.hbr.idata.posterior['sigma_intercept_mu'].to_numpy().mean()
        offsets = model.hbr.idata.posterior['offset_intercept_mu'].to_numpy().mean(axis=(0, 1))

        centered_Y_train = np.zeros_like(this_Y_train)
        for i, be in enumerate(np.unique(Z_train)):
            this_offset_intercept = offsets[i]
            idx = (Z_train == be).all(1)
            centered_Y_train[idx] = this_scaled_Y_train[idx] - sigma_intercept_mu * this_offset_intercept

        # Setup figure or use provided axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = None  # we'll return nothing in this case

        ytrain_inv = outscaler.inverse_transform(centered_Y_train[train_sex_idx, None])
        ax.scatter(inscaler.inverse_transform(this_scaled_X_train)[train_sex_idx, 0], ytrain_inv,
                   alpha=0.1, s=12, color=cols[0])

        be_map = np.unique(Z_train)
        difX = np.max(this_scaled_X_train[:, 0]) - np.min(X_train[:, 0])
        min0 = np.min(this_scaled_X_train[:, 0]) + 0.01 * difX
        max0 = np.max(this_scaled_X_train[:, 0]) - 0.01 * difX
        sex = np.unique(this_scaled_X_train[:, 1])[selected_sex_id]
        synthetic_X0 = np.linspace(min0, max0, 200)[:, None]
        synthetic_X = np.concatenate((synthetic_X0, np.full(synthetic_X0.shape, sex)), axis=1)

        model_be_long = np.repeat(np.array(be_map[model_be]), synthetic_X.shape[0])
        q = model.get_mcmc_quantiles(synthetic_X, model_be_long) - sigma_intercept_mu * offsets[model_be]
        q = outscaler.inverse_transform(q).T
        x = inscaler.inverse_transform(synthetic_X)

        ran = np.arange(-3, 4)
        for ir, r in enumerate(ran):
            if r == 0:
                ax.plot(x[:, 0], q[:, ir], color='black')
            elif abs(r) == 3:
                ax.plot(x[:, 0], q[:, ir], color='black', alpha=0.6, linestyle="--", linewidth=1)
            else:
                ax.plot(x[:, 0], q[:, ir], color='black', alpha=0.6, linewidth=1)

        lmap = {'blr': 'W-BLR', 'SHASHo': '$\mathcal{S}_o$', 'SHASHb_1': '$\mathcal{S}_{b1}$',
                'SHASHb_2': '$\mathcal{S}_{b2}$', 'Normal': '$\mathcal{N}$'}

        suffix = "estimate"
        ax.set_title(f"{lmap[l]}: {f_idx},{feature}", fontsize=10)
        ax.set_xlabel('Age', fontsize=8)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.grid(linestyle=":", linewidth=1, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.yaxis.offsetText.set_fontsize(8)

        del model, this_Y_train, this_scaled_X_train, this_scaled_Y_train
        gc.collect()

        return fig  # Only used if ax is None

    except Exception as e:
        print(f"Could not plot {feature}: {e}")
        if ax:
            ax.set_title(f"{feature} (error)", fontsize=10)
            ax.axis('off')
        return None


# === Generate plots: 8 per PDF page ===
output_dir = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/quantile_batches"
os.makedirs(output_dir, exist_ok=True)

batch_size = 8
ncols, nrows = 2, 4

for i in range(0, len(features), batch_size):
    batch = features[i:i + batch_size]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 14), dpi=100)
    axes = axes.flatten()

    for j, feature in enumerate(batch):
        ax = axes[j]
        plot_MAP_quantiles(feature=feature, l='Normal', atlas='DES', ax=ax)

    # Hide unused axes
    for k in range(len(batch), len(axes)):
        axes[k].axis("off")

    plt.tight_layout()

    # Save each batch as its own PDF file
    pdf_filename = os.path.join(output_dir, f"quantile_batch_{i // batch_size + 1}.pdf")
    fig.savefig(pdf_filename, dpi=100, bbox_inches="tight")
    plt.close(fig)
