# creating objective functions SEPARATELY for 3 different MPOs: fexofenadine, ranolazine, and osimertinib objectives
from pprint import pprint

import numpy as np
import pandas as pd

from gp_mobo.optimiser import evaluate_objectives
from tdc_oracles_modified import Oracle

# Load GuacaMol dataset
guacamol_dataset_path = "guacamol_dataset/guacamol_v1_train.smiles"
guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
ALL_SMILES = guacamol_dataset["smiles"].tolist()[:10_000]
known_smiles = ALL_SMILES[:50]
print("Known SMILES:")
pprint(known_smiles)

# Create "oracles" for FEXOFENADINE objectives
TPSA_ORACLE = Oracle("tpsa_score_single")
LOGP_ORACLE = Oracle("logp_score_single")
FEXOFENADINE_SIM_ORACLE = Oracle("fex_similarity_value_single")

# Create "oracles" for OSIMERTINIB objectives
OSIM_TPSA_ORACLE = Oracle("osimertinib_tpsa_score")
OSIM_LOGP_ORACLE = Oracle("osimertinib_logp_score")
OSIM_SIM_V1_ORACLE = Oracle("osimertinib_similarity_v1_score")
OSIM_SIM_V2_ORACLE = Oracle("osimertinib_similarity_v2_score")

# Create "oracles" for RANOLAZINE objectives
RANOL_TPSA_ORACLE = Oracle("ranolazine_tpsa_score")
RANOL_LOGP_ORACLE = Oracle("ranolazine_logp_score")
RANOL_SIM_ORACLE = Oracle("ranolazine_similarity_value")
RANOL_FLUORINE_ORACLE = Oracle("ranolazine_fluorine_value")

# 1st MPOs to investigate as baseline
FEXOFENADINE_MPO_ORACLE = Oracle("fexofenadine_mpo")
# 2nd and 3rd MPO to investigate as baseline
OSIMERTINIB_MPO_ORACLE = Oracle("osimertinib_mpo")
RANOLAZINE_MPO_ORACLE = Oracle("ranolazine_mpo")


def evaluate_fex_objectives(smiles_list: list[str]) -> np.ndarray:
    return evaluate_objectives(smiles_list, [TPSA_ORACLE, LOGP_ORACLE, FEXOFENADINE_SIM_ORACLE])


def evaluate_osim_objectives(smiles_list: list[str]) -> np.ndarray:
    return evaluate_objectives(
        smiles_list, [OSIM_TPSA_ORACLE, OSIM_LOGP_ORACLE, OSIM_SIM_V1_ORACLE, OSIM_SIM_V2_ORACLE]
    )


def evaluate_ranol_objectives(smiles_list: list[str]) -> np.ndarray:
    return evaluate_objectives(
        smiles_list, [RANOL_TPSA_ORACLE, RANOL_LOGP_ORACLE, RANOL_SIM_ORACLE, RANOL_FLUORINE_ORACLE]
    )


# """
# SINGLE MPO OBJECTIVES <- objectives are not separated here:
# 1) fexofenadine MPO
# 2) ranolazine MPO
# 3) osimertinib MPO
# """


def evaluate_fex_MPO(smiles_list: list[str]) -> np.ndarray:
    return evaluate_objectives(smiles_list, [FEXOFENADINE_MPO_ORACLE])


def evaluate_ranol_MPO(smiles_list: list[str]) -> np.ndarray:
    return evaluate_objectives(smiles_list, [RANOLAZINE_MPO_ORACLE])


def evaluate_osim_MPO(smiles_list: list[str]) -> np.ndarray:
    return evaluate_objectives(smiles_list, [OSIMERTINIB_MPO_ORACLE])
