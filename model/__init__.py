from model.configs import GraphConfig, GraphLlavaConfig, ProjectorConfig, MoEConfig
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoConfig, AutoTokenizer
import torch
from arguments import ModelArguments, TrainingArguments
from utils import no_init_weights
from model.modeling_llava import GraphLlavaForConditionalGeneration
from loggers import WrappedLogger

logger = WrappedLogger(__name__)

def find_linear_without_moe_gates(model) -> list:
    """Find all linear modules except for graph_tower, mm_projector and lm_head

    Args:
        model (nn.Module): Model

    Returns:
        list: list of found modules
    """
    cls = torch.nn.Linear
    lora_module_names = list()
    exception_list = (
        "graph_tower", 
        "mm_projector", 
        "lm_head", 
        "deepspeed_moe", 
        "mlp.mlp", 
        "mlp.coefficient", 
        "moe_gate"
    )
    for name, module in model.named_modules():
        if isinstance(module, cls) and all([exception not in name for exception in exception_list]):
            lora_module_names.append(name)
            
    return lora_module_names

def create_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        use_mlp=model_args.use_mlp,
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        enable_apple_loss=model_args.enable_apple_loss
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.load_projector(model_args.pretrain_mm_mlp_adapter)
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. create moe model
    moe_config = MoEConfig(
        moe_mode = model_args.moe_mode,
        moe_layers_idx=model_args.moe_layers_idx,
        ep_size=model_args.ep_size,
        num_experts=model_args.num_experts,
        top_k_experts=model_args.top_k_experts,
        capacity_factor=model_args.capacity_factor,
        eval_capacity_factor=model_args.eval_capacity_factor,
        min_capacity=model_args.min_capacity,
        use_residual=model_args.use_residual,
        router_aux_loss_coef=model_args.router_aux_loss_coef,
        moe_class=model_args.moe_class,
        norm_topk_prob=model_args.norm_topk_prob
    )
    
    model.config.moe_enable = True,
    model.config.text_config.moe_enable = True
    model.config.moe_config = moe_config
    training_args.moe_enable = model_args.moe_enable = True
    
    if torch.cuda.is_available():
        logger.info("Moving to CUDA for faster creation", on_rank0=True)
        model.to("cuda")
        
    model.replace_mlp_with_moe()
    model.to("cpu")
    torch.cuda.empty_cache()
    
    # 5. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_linear_without_moe_gates(model),  # do not add lora to any MoE layers, but any layer except that
        use_rslora=training_args.use_rslora,
        use_learnable_alpha=training_args.use_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        logger.info("Moving to cuda for faster warping...", on_rank0=True)
        model.to("cuda")
        
    logger.info("Adding LoRA adapters...", on_rank0=True)
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
    
    # 6. activate moe gate
    for name, module in model.named_modules():
        if 'moe_gate' in name:
            module.requires_grad_(True)
    
    return tokenizer, model
