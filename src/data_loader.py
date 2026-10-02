from pathlib import Path
import h5py
import pandas as pd


# ==================================================
# DATA PATH
# ==================================================

RAW_DATA = Path("../data/raw")


# ==================================================
# DATASET COLUMN DEFINITIONS
# ==================================================

A_COLUMNS = [
    "unit",
    "cycle",
    "Fc",
    "hs"
]


T_COLUMNS = [
    "fan_eff_mod",
    "fan_flow_mod",
    "LPC_eff_mod",
    "LPC_flow_mod",
    "HPC_eff_mod",
    "HPC_flow_mod",
    "HPT_eff_mod",
    "HPT_flow_mod",
    "LPT_eff_mod",
    "LPT_flow_mod"
]


W_COLUMNS = [
    "alt",
    "Mach",
    "TRA",
    "T2"
]


X_S_COLUMNS = [
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf"
]


X_V_COLUMNS = [
    "T40",
    "P30",
    "P45",
    "W21",
    "W22",
    "W25",
    "W31",
    "W32",
    "W48",
    "W50",
    "SmFan",
    "SmLPC",
    "SmHPC",
    "phi"
]


# ==================================================
# DATASET LOADER
# ==================================================

def load_dataset(file_name):

    file_path = RAW_DATA / file_name

    with h5py.File(file_path, "r") as f:

        data = {
            "A": pd.DataFrame(
                f["A_dev"][:],
                columns=A_COLUMNS
            ),

            "T": f["T_dev"][:],

            "W": f["W_dev"][:],

            "X_s": f["X_s_dev"][:],

            "X_v": f["X_v_dev"][:],

            "Y": f["Y_dev"][:],
        }

    return data