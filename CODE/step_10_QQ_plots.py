#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:00:07 2026

@author: johbay
QQ plots are generated as part of step_9_2_ADNI_OASIS_z_gain_revision. This script collates them into a large png
"""

from PIL import Image
from pathlib import Path
import math

atlas="DK"
dataset="OASIS"

files = sorted(Path("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/QQ_plots").glob(f"HC_QQ*{dataset}_{atlas}*revision.png"))    # your PNGs, in order
ncols = 5                                        # set columns; rows computed from count
nrows = math.ceil(len(files) / ncols)

imgs = [Image.open(f) for f in files]
w = max(im.width for im in imgs)                 # cell size = largest image
h = max(im.height for im in imgs)

grid = Image.new("RGB", (ncols * w, nrows * h), "white")
for i, im in enumerate(imgs):
    r, c = divmod(i, ncols)
    grid.paste(im, (c * w, r * h))

grid.save(f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/combined_HC_QQ_plots_{dataset}_{atlas}.png")


#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math
from PIL import Image
from pathlib import Path


atlases = ["SC", "DK", "DES"]
datasets = ["ADNI", "OASIS"]

for atlas in atlases:
    for dataset in datasets:
        files = sorted(Path("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/QQ_plots")
                       .glob(f"HC_QQ*{dataset}_{atlas}*revision.png"))
        
        A4_W, A4_H = 2480, 3508        # A4 portrait at 300 DPI
        ncols, nrows = 5, 10
        per_page = ncols * nrows       # 35 plots per page
        margin = 40
        
        cell_w = (A4_W - 2 * margin) // ncols
        cell_h = (A4_H - 2 * margin) // nrows
        
        for page, start in enumerate(range(0, len(files), per_page)):
            chunk = files[start:start + per_page]
            page_img = Image.new("RGB", (A4_W, A4_H), "white")
            for i, f in enumerate(chunk):
                im = Image.open(f)
                im.thumbnail((cell_w, cell_h))                       # fit cell, keep aspect ratio
                r, c = divmod(i, ncols)
                x = margin + c * cell_w + (cell_w - im.width) // 2    # center in cell
                y = margin + r * cell_h + (cell_h - im.height) // 2
                page_img.paste(im, (x, y))
            outpath = (f"/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/imgs/"
                       f"combined_HC_QQ_plots_{dataset}_{atlas}_page{page + 1}.png")
            page_img.save(outpath, dpi=(300, 300))
            print(f"page {page + 1}: {len(chunk)} plots → {outpath}")