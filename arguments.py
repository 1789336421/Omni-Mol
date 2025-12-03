from dataclasses import dataclass, field

from dataclasses import dataclass, field
from typing import Optional, List, Any
import transformers

@dataclass
class ModelArguments:
    # LLM args
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    base_model: str = field(default="checkpoints/phi3")
    language_backbone_name: str = field(default="checkpoints/phi3")
    # Graph args
    graph_tower: Optional[str] = field(default=None)
    gin_num_layers: int = field(default=5)
    gin_hidden_dim: int = field(default=300)
    drop_ratio: float = field(default=0.1)
    graph_pooling: str = field(default='mean')
    graph_init_checkpoint: Optional[str] = field(default=None)
    # projector args
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    use_mlp:bool = field(default=False)
    # MoE args
    moe_enable: bool = field(default=False)
    moe_class: str = field(default="deepseek")
    moe_mode: str = field(
        default="sparse",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["first_half", "second_half", "second_quarter", "sparse", "dense"],
        },
    )
    moe_layers_idx: Optional[List[int]] = field(default=None, metadata={"help": "where to place moe layers."})
    ep_size: int = field(default=1)
    num_experts: int = field(default=3, metadata={"help": "number of experts for each moe layer."})
    top_k_experts: int = field(
        default=2,
        metadata={
            "help": "Top-k experts to deal with tokens.",
            "choices": [1, 2],
        },
    )
    capacity_factor: float = field(default=1.)
    eval_capacity_factor: float = field(default=2.)
    min_capacity: int = field(default=0)
    use_residual: bool = field(default=False)
    router_aux_loss_coef: float = field(default=0.01)
    enable_apple_loss: bool = field(default=False)
    norm_topk_prob: bool = field(default=False)


@dataclass
class DataArguments:
    data_type: str = field(default="supervised")
    data_path: str = field(default=None, metadata={"help": "Path to the training data.(.pkl)"})
    is_training: bool = field(default=True)
    add_selfies: bool = field(default=True)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # misc
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # Model specific
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # LoRA settings
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"
    use_rslora: bool = field(default=False)
    use_alpha: bool = field(default=True)
    resume_from_lora: str = field(default=None)
    # training recipes
    training_recipe: str = field(
        default="stage1",
        metadata={"help": "Choose from stage1, stage2, stage3"}
        )
    stop_epoch: int = field(default=10)
    total_size: int = field(
        default=3e5
    )
