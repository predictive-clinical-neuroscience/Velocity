# Velocity

Modelling scripts for velocity normative models.
Currently being updated as part of revision.

<img width="800" height="523" alt="image" src="https://github.com/user-attachments/assets/ad8c7bdd-965a-4781-8dcd-3c8274d8a9e0" />



## Repository layout

| path | contents |
| --- | --- |
| `CODE/` | full analysis pipelien for paper |
| `VELOCITY/` | the `z_gain` package and ready-to-run templates for computing z-gains on any longitudinal dataset |


## Computing z-gains

A **z-gain** measures how much a subject's normative-model z-score changed between two
timepoints, corrected for how strongly z-scores at those two ages are expected to
correlate, based on a reference population:

$$\text{z-gain} = \frac{Z_2 - r\,Z_1}{\sqrt{1 - r^2}}$$

where $r$ is the expected correlation between z-scores at $age_1$ and $age_2$. Without
that correction, a subject who merely regresses toward the mean looks like they changed.

`VELOCITY/zgain` packages this so it can be applied to any longitudinal dataset, not
just the ones in this study.

### Install

```bash
pip install -e VELOCITY/zgain
```

Requires Python 3.9+, numpy and pandas.

### Getting started

Three entry points, in increasing order of how much you want to automate:

| | |
| --- | --- |
| `VELOCITY/run_velocity_template.ipynb` | **Start here.** A notebook that runs end to end on synthetic data the moment you open it, then takes your own files once you set `USE_DEMO_DATA = False`. |
| `VELOCITY/run_velocity_template.py` | The same workflow as a command-line script, for batching over many measures. Run `--demo` first to check the install. |
| `z-gain` | A console script installed with the package, for a single measure. |

```bash
python VELOCITY/run_velocity_template.py --demo          # synthetic data, no inputs needed
python VELOCITY/run_velocity_template.py --data my_z.csv --velocity R.pkl \
    --measure Right-Hippocampus --output-dir results/
```

Or from Python:

```python
from z_gain import compute_all_z_gains, load_velocity_matrix, validate_zgain_frame

R = load_velocity_matrix("velocity_objects.pkl")
print(validate_zgain_frame(df, "Right-Hippocampus", R=R).summary())
gains = compute_all_z_gains(df, R, "Right-Hippocampus")
```

### Input data

A table in **long** format, one row per subject per visit, with four columns (the names
are configurable):

| role | default name | requirement |
| --- | --- | --- |
| subject id | `ID_subject` | repeated across that subject's visits |
| age | `age` | numeric. **Used as an integer index** into the velocity matrix, so ages are effectively rounded down to whole years |
| visit id | `ID_visit` | only used to order rows that share an age |
| measure | *you name it* | the **z-score** for the measure, not the raw volume or thickness |

Any other columns (diagnosis, sex, site) ride along and are ignored.

```
ID_subject  age  ID_visit  Right-Hippocampus
  sub-0001   65         1              -1.88
  sub-0001   66         2              -1.70
  sub-0001   68         3              -1.86
  sub-0002   71         1               0.42
```

### The velocity matrix

You also need an age-by-age correlation matrix: a pickle holding either the array
itself, or a dict with an `A_sparse_predict` key. Sparse matrices are densified
automatically.

For this study these are produced by `make_velocity_plots` in
`CODE/utilities_thrive_new_version.py`, which writes one `velocity_objects.pkl` per
measure. To use the package on your own data you need the equivalent for your own
normative model. The matrices here are banded, so `r` is exactly 0 beyond a fixed age
gap and those pairs are dropped.

### Output

One row per **pair of timepoints**, not per visit. A subject with 3 distinct ages gives
3 rows; a subject with a single visit gives none.

| column | meaning |
| --- | --- |
| `id` | subject |
| `age1`, `age2` | the two ages, always `age2 > age1` |
| `t1_index`, `t2_index` | positions within that subject's visit sequence |
| `z_gain` | the corrected change |
| `Z1`, `Z2` | the input z-scores |

### Validate before you trust a result

`compute_all_z_gains` catches per-pair errors and logs a warning, so a malformed input
yields a *smaller* result rather than an error. Pairs are dropped when the age is NaN or
falls outside the matrix, or when $r < 0.3$. NaN z-scores are worse: they raise nothing
and flow through as NaN gains.

`validate_zgain_frame` reports all of this up front:

```
subjects=557, rows=2350, expected gains=1711
WARNING 908 row(s) repeat an age within a subject; only the first (by visit order) is kept
WARNING 144 subject(s) have fewer than 2 distinct ages and contribute no gains
```

Two patterns are worth understanding if you see them. **Repeated ages within a subject**
mean your ages collapse to the same whole year, and only the first survives — if your
visits are months apart this can discard a large fraction of your data. **Pairs with
`r < 0.3`** are too far apart in age for the velocity matrix to relate them.
