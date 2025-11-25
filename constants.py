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

MODEL_CLS_MAP = {
    "llama": CustomLlamaForCausalLM,
}

MODEL_CONF_MAP = {
    "llama": LlamaConfig
}