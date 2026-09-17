"""
Central configuration for the whole package.

Corresponds to notebook Section 1.2 and 1.3 (Configuration, Random
Seeds & Target). Every other module should import constants from
here instead of hard-coding them again -- this is the single source
of truth for reproducibility (random_state, target column name,
data path, CV strategy parameters, etc.).

Fill in:
    - RANDOM_STATE
    - TARGET
    - DATA_PATH (consider making this an argument/env var instead of
      a hard-coded Windows path, unlike the original notebook -- this
      matters for portability once this becomes a real package)
    - CV_N_SPLITS, CV_N_REPEATS (Section 8, RepeatedKFold config)
    - TEST_SIZE (Section 5 train_test_split)
    - Any sklearn global config (e.g. set_config(transform_output=...),
      see Section 1.2 -- decide whether this belongs here as a
      side-effecting call on import, or should be called explicitly
      by the training entrypoint. A config module silently mutating
      global sklearn state on import is a debatable design choice --
      make it deliberate, not accidental.)
"""

# RANDOM_STATE = ...
# TARGET = ...
# DATA_PATH = ...
# TEST_SIZE = ...
# CV_N_SPLITS = ...
# CV_N_REPEATS = ...
