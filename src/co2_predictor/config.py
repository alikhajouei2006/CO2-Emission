"""
Central configuration for the whole package.

Corresponds to notebook Section 1.2 and 1.3 (Configuration, Random
Seeds & Target). Every other module should import constants from
here instead of hard-coding them again -- this is the single source
of truth for reproducibility (random_state, target column name,
data path, CV strategy parameters, etc.).
"""

from pathlib import Path

RANDOM_STATE = 42
TARGET = 'CO2 Emissions(g/km)'
DATA_PATH = Path(r"\CO2-Emission\data\raw\CO2 Emissions_Canada.csv")
TEST_SIZE = 20 / 100
CV_N_SPLITS = 3
CV_N_REPEATS = 5
