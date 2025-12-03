from data_pipe.dataset_profiles import (
    ForwardPredDataset, 
    RetrosynDataset, 
    ReagentPredDataset, 
    MolcapDataset,
    PropertyPredDataset,
    SolventPredDataset,
    CatalystPredDataset,
    YieldRegressionDataset,
    ExpProcedurePrediction,
    TPSAPrediction,
    WeightPrediction,
    DescriptionQA,
    LogPPrediction,
    IUPAC,
    TextGuidedMolGen,
    MolEditing
)
from model.modeling_llama import CustomLlamaForCausalLM
from transformers.models.llama import LlamaConfig

# Registration of datasets
TASK_MAP = {
    "forward": ForwardPredDataset,
    "reagent": ReagentPredDataset,
    "retrosynthesis": RetrosynDataset,
    "molcap": MolcapDataset,
    "homolumo": PropertyPredDataset,
    "solvent": SolventPredDataset,
    "catalyst": CatalystPredDataset,
    "yield_BH": YieldRegressionDataset,
    "yield_SM": YieldRegressionDataset,
    "experiment": ExpProcedurePrediction,
    "tpsa": TPSAPrediction,
    "weight": WeightPrediction,
    "dqa": DescriptionQA,
    "logp": LogPPrediction,
    "iupac": IUPAC,
    'textguidedmolgen': TextGuidedMolGen,
    "molediting": MolEditing,
}

MAP2FILE = {
    "forward": "forward_prediction",
    "reagent": "reagent_prediction",
    "retrosynthesis": "retrosynthesis",
    "molcap": "molcap",
    "homolumo": "property_prediction",
    "solvent": "solvent_pred",
    "catalyst": "catalyst_pred",
    "yield": "yield_regression",
    "experiment": "exp_procedure_pred",
    "tpsa": "TPSA",
    "weight": "Molecular_Weight",
    "dqa": "DescriptionQA",
    "logp": "LogP",
    "iupac": "IUPAC2SELFIES",
    'textguidedmolgen': "text_guided_mol_generation",
    "molediting": "molecule_editing"
}

MODEL_CLS_MAP = {
    "llama": CustomLlamaForCausalLM,
}

MODEL_CONF_MAP = {
    "llama": LlamaConfig
}