# ruff: noqa
# Add additional objectives for the MPO methods <- to test
# Add additional objectives for the MPO methods <- to test
from .oracle.oracle import fex_similarity_value_single  # fexofenadine separated MPO
from .oracle.oracle import logp_score_single  # fexofenadine separated MPO
from .oracle.oracle import osimertinib_logp_score  # osimertinib separated MPO
from .oracle.oracle import osimertinib_similarity_v1_score  # osimertinib separated MPO
from .oracle.oracle import osimertinib_similarity_v2_score  # osimertinib separated MPO
from .oracle.oracle import osimertinib_tpsa_score  # osimertinib separated MPO
from .oracle.oracle import ranolazine_fluorine_value  # ranolazine separated MPO
from .oracle.oracle import ranolazine_logp_score  # ranolazine separated MPO
from .oracle.oracle import ranolazine_similarity_value  # ranolazine separated MPO
from .oracle.oracle import ranolazine_tpsa_score  # ranolazine separated MPO
from .oracle.oracle import tpsa_score_single  # fexofenandine separated MPO
from .oracle.oracle import (
    SA,
    PyScreener_meta,
    Score_3d,
    Vina_3d,
    Vina_smiles,
    albuterol_similarity,
    amlodipine_mpo,
    aripiprazole_similarity,
    askcos,
    celecoxib_rediscovery,
    cyp3a4_veith,
    deco_hop,
    drd2,
    fexofenadine_mpo,
    gsk3b,
    ibm_rxn,
    isomer_meta,
    isomers_c7h8n2o2,
    isomers_c9h10n2o2pf2cl,
    isomers_c11h24,
    jnk3,
    median1,
    median2,
    median_meta,
    mestranol_similarity,
    molecule_one_retro,
    osimertinib_mpo,
    penalized_logp,
    perindopril_mpo,
    qed,
    ranolazine_mpo,
    rediscovery_meta,
    scaffold_hop,
    similarity_meta,
    sitagliptin_mpo,
    sitagliptin_mpo_prev,
    thiothixene_rediscovery,
    troglitazone_rediscovery,
    valsartan_smarts,
    zaleplon_mpo,
    zaleplon_mpo_prev,
)
