# Changelog

<!--next-version-placeholder-->

## Unreleased

- Add `random_state` to `arf()` and `forge()` for reproducible fitting and sampling; `None` keeps the previous behaviour (numpy's global random state)
- Fix `forge()` failing with "probabilities do not sum to 1" when leaves hold a single observation: leaf coverage is renormalised within each tree (#4)
- Fix `forde(oob=True)`, which referenced a missing attribute; out-of-bag rows now come from the public `estimators_samples_`
- Drop the deprecated `copy=` argument of `DataFrame.set_axis` (pandas 3)
- Fix compatibility with NumPy 2 and pandas 3 (`np.in1d` → `np.isin`, positional Series access via `.iloc`)
- Migrate packaging from `setup.py` to `pyproject.toml` (uv / hatchling)
- Fix the twomoons example notebook: define a held-out `df_test` via `train_test_split` and drop a redundant resampling step

## v0.1.0 (17/04/2023)

- First release of `arfpy`

## v0.1.1 (22/09/2023)

- Update License to MIT
- Allow for variables with zero standard deviation in terminal nodes 
- modify code to lighten dependency requirements: allow for numpy >=1.24 and pandas >= 2.0
