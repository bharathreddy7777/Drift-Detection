# ----- Model Parameters -----
N_ESTIMATORS = 30
RANDOM_STATE = 42
MAX_DEPTH = 15
MAX_SAMPLES = 0.5        # use 50% of data per tree to save memory
MAX_TRAIN_ROWS = 20000   # cap training data to avoid OOM

# ----- Default Experiment Settings -----
DEFAULT_N_BATCHES = 10
DEFAULT_DRIFT_START_RATIO = 0.3  # drift starts after 30% of batches
DEFAULT_DRIFT_STRENGTH = 0.4
DEFAULT_CONCEPT_DRIFT_RATIO = 0.4
DEFAULT_WINDOW_SIZE = 5

# ----- Drift Detection Thresholds -----
KS_THRESHOLD = 0.05
PSI_THRESHOLD = 0.2
ERROR_RATE_THRESHOLD = 0.15
