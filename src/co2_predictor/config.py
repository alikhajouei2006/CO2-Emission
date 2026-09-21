from pathlib import Path
from sklearn import set_config
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CO2 Emissions_Canada.csv"
RANDOM_STATE = 42
TARGET = 'CO2 Emissions(g/km)'
TEST_SIZE = 20 / 100
CV_N_SPLITS = 5
CV_N_REPEATS = 3
N_ITER = 50

def initial_configurations() -> None:    
    set_config(transform_output="pandas")
    warnings.filterwarnings("ignore")