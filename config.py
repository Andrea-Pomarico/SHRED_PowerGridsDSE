# =============================================================================
# config.py  –  All user-facing parameters for the DSE experiment
# =============================================================================
# Edit ONLY this file to change paths, fault scenarios, or hyper-parameters.
# Every other module imports from here.

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT           = "Data"           # root folder that contains N/ and N_1/

# ── Topology ──────────────────────────────────────────────────────────────────
TOPOLOGY_MODE       = "N"              # "N" (all lines) or "N_1" (one line out)
LINE_OUT_OF_SERVICE = "Line 08 - 09"  # used only when TOPOLOGY_MODE == "N_1"

# ── Scenario lists ────────────────────────────────────────────────────────────
LIST_TEMPO = ["2", "3", "4", "5", "6", "7", "8"]   # fault clearing times

LIST_FAULT = [
    "Line 01 - 02", "Line 01 - 39", "Line 02 - 03", "Line 02 - 25",
    "Line 03 - 04", "Line 03 - 18", "Line 04 - 05", "Line 04 - 14",
    "Line 05 - 06", "Line 05 - 08", "Line 06 - 07", "Line 06 - 11",
    "Line 07 - 08",
]

# ── Power-system dimensions ───────────────────────────────────────────────────
N_BUS = 39
N_GEN = 10

# ── Noise injection ───────────────────────────────────────────────────────────
ADD_NOISE      = True   # False → use clean data
NOISE_STD_FRAC = 0.15   # σ = NOISE_STD_FRAC × std(variable)
NUM_NOISY_VARS = 117    # how many columns to corrupt (first N columns)
NOISE_SEED     = 42     # random seed for reproducibility

# ── PyShred / DataManager ─────────────────────────────────────────────────────
LAGS       = 35
TRAIN_SIZE = 0.8
VAL_SIZE   = 0.1
TEST_SIZE  = 0.1

# Stationary sensor indices – one tuple per sensor
# (Group 2 – 10 buses, as used in the paper)
STATIONARY_SENSORS = [(i,) for i in [
    3, 4, 5,   6, 7, 8,   12, 13, 14,  24, 25, 26,
    36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68,
    78, 79, 80, 81, 82, 83,
]]

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 10
BATCH_SIZE = 1024
LR         = 1e-4

# ── Visualisation ─────────────────────────────────────────────────────────────
TEST_WINDOW  = (16000, 18100)   # time window [start, end] for test plots
TRAIN_WINDOW = (144000, 145500) # time window [start, end] for train plots
VARS_TO_PLOT = range(138)       # change to e.g. [0, 1, 40, 76] for a subset
