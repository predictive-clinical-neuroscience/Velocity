#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 14:23:19 2025

@author: johbay
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# --------- CONFIG ----------
DIR_A = Path("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA/DK")
DIR_B = Path("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/DATA_pre_processed")
EXTS = {".pkl", ".csv"}

RTOL = 1e-6
ATOL = 1e-8
# --------------------------


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compare(a, b, path="root"):
    """Return (ok: bool, msg: str|None). Stops at first difference."""

    # --- pandas ---
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        if not (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)):
            return False, f"{path}: type mismatch {type(a)} vs {type(b)}"
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=RTOL, atol=ATOL)
            return True, None
        except AssertionError as e:
            return False, f"{path}: DataFrame differs ({str(e).splitlines()[0]})"

    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        if not (isinstance(a, pd.Series) and isinstance(b, pd.Series)):
            return False, f"{path}: type mismatch {type(a)} vs {type(b)}"
        try:
            pd.testing.assert_series_equal(a, b, check_exact=False, rtol=RTOL, atol=ATOL)
            return True, None
        except AssertionError as e:
            return False, f"{path}: Series differs ({str(e).splitlines()[0]})"

    # --- numpy arrays ---
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False, f"{path}: type mismatch {type(a)} vs {type(b)}"
        if a.shape != b.shape:
            return False, f"{path}: shape differs {a.shape} vs {b.shape}"

        # float/complex: tolerant; others: exact
        if a.dtype.kind in {"f", "c"} or b.dtype.kind in {"f", "c"}:
            ok = np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
            if ok:
                return True, None
            diff_mask = ~np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
            first = tuple(np.argwhere(diff_mask)[0]) if diff_mask.any() else None
            return False, f"{path}: ndarray values differ (first diff at {first})"
        else:
            if np.array_equal(a, b, equal_nan=True):
                return True, None
            return False, f"{path}: ndarray values differ"

    # --- containers (in case pickles contain nested stuff) ---
    if type(a) != type(b):
        return False, f"{path}: type mismatch {type(a)} vs {type(b)}"

    if isinstance(a, dict):
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb:
            only_a = sorted(ka - kb)
            only_b = sorted(kb - ka)
            return False, f"{path}: dict keys differ (only in A: {only_a[:10]} | only in B: {only_b[:10]})"
        for k in a:
            ok, msg = compare(a[k], b[k], path=f"{path}.{k!r}")
            if not ok:
                return False, msg
        return True, None

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, f"{path}: length differs {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, msg = compare(x, y, path=f"{path}[{i}]")
            if not ok:
                return False, msg
        return True, None

    # --- scalars / fallback ---
    if isinstance(a, float):
        if not np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            return False, f"{path}: float differs {a} vs {b}"
        return True, None

    if a != b:
        return False, f"{path}: value differs {a!r} vs {b!r}"
    return True, None


def main():
    files_a = {p.name: p for p in DIR_A.glob("*") if p.is_file() and p.suffix in EXTS}
    files_b = {p.name: p for p in DIR_B.glob("*") if p.is_file() and p.suffix in EXTS}

    common = sorted(set(files_a) & set(files_b))
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))

    print(f"Folder A: {DIR_A.resolve()}")
    print(f"Folder B: {DIR_B.resolve()}")
    print(f"Common filenames: {len(common)}")
    print(f"Only in A: {len(only_a)}")
    print(f"Only in B: {len(only_b)}\n")

    for name in common:
        pa, pb = files_a[name], files_b[name]
        print(f"▶ {name}")
        try:
            if pa.suffix == ".csv":
                a = load_csv(pa)
                b = load_csv(pb)
                
            elif pa.suffix == ".pkl":   # .pkl
                a = load_pickle(pa)
                b = load_pickle(pb)
            else:
                pass
            
            ok, msg = compare(a, b)
            if ok:
                print("   ✔ identical")
            else:
                print(f"   ✖ different: {msg}")
                print(f"=== {DIR_A} ===")
                print(a.head())
                print(a.shape)

                print(f"\n=== {DIR_B} ===")
                print(b.head())
                print(b.shape)
          
        except Exception as e:
            print(f"   ⚠ error: {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()