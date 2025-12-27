"""
@ 2025 Omni-Mol Project

Inference script, supports parallel inference if you have multiple GPUs
"""
import os
import torch
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from loggers import WrappedLogger
from model.modeling_llava import GraphLlavaForConditionalGeneration
from model.configs import GraphLlavaConfig
from utils import no_init_weights, save_json
from peft import PeftModel
from typing import Sequence, Dict, Any
from torch_geometric.data import Data
from data_pipe import conversation_lib
from constants import TASK_MAP
from datetime import timedelta
from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizer,
    GenerationConfig,
    set_seed
)
from metrics import (
    calculate_reaction_metrics,
    calculate_r2,
    calculate_homolumo_metrics,
    calculate_mae_with_text,
    calculate_text_scores,
    calculate_moledit_metrics
)


LLAMA3_EOS = "<|eot_id|>"

logger = WrappedLogger(__name__)


def setup_distributed(rank: int, world_size: int):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # An unused port
    # Initialize the process group
    # To avoid NCCL timeout
    TIMEOUT = timedelta(hours=24)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, 
        device_id=torch.device(f"cuda:{rank}"),
        timeout=TIMEOUT
    )
    # Pin the current process to a specific GPU
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()
    torch.cuda.empty_cache()

def split_data(data: list[dict], world_size: int):
    """
    Splits a list of data into 'world_size' chunks, handling uneven splits.
    Uses numpy.array_split for a simple and robust split.
    """
    # np.array_split will create as-equal-as-possible chunks.
    # It returns a list of numpy arrays, so we convert them back to lists.
    return [arr.tolist() for arr in np.array_split(data, world_size)]

def _convert_dict_to_Data(data_dict: Dict) -> Data:
    return Data(
        x=torch.asarray(data_dict['node_feat']),
        edge_attr=torch.asarray(data_dict['edge_feat']),
        edge_index=torch.asarray(data_dict['edge_index']),
    )

def data_processing(
    instance: Sequence[Dict], 
    tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    input_ids_list = [instance["input_ids"]]
    gt_list = [instance["gt"]]
    prompt_list = [instance["prompt"]]

    batch_input = tokenizer.pad(
        {"input_ids": input_ids_list}, 
        padding=True,
        return_tensors="pt"
    )

    payload = {
        'input_ids': batch_input["input_ids"][:, :tokenizer.model_max_length],
        'gt': gt_list,  # This is a list: [instance["gt"]]
        'prompt': prompt_list, # This is a list: [instance["prompt"]]
        'attention_mask': batch_input["attention_mask"][:, :tokenizer.model_max_length],
    }
    
    if 'graphs' in instance:
        graph_batch = []
        if instance["graphs"] is not None:
            graph_batch.append(_convert_dict_to_Data(instance["graphs"]))
        else:
            graph_batch.append(None)
        
        # This will be a list with one item: [Data(...)] or [None]
        payload["graphs"] = graph_batch
        
    return payload

def build_model(
    model_path: str,
    language_backbone: str,
    graph_path: str
) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    config = GraphLlavaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    generation_config = GenerationConfig.from_pretrained(language_backbone)

    attn_impl = "flash_attention_2"
    try:
        config._attn_implementation = attn_impl
        config.text_config._attn_implementation = attn_impl
        config.torch_dtype = torch.bfloat16
        config.text_config.torch_dtype = torch.bfloat16
        with no_init_weights():
            model = GraphLlavaForConditionalGeneration(config)
    except Exception as e:
        logger.warning(f"Failed to initialize with Flash Attention 2 ({e}), falling back to 'eager'.")
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        config.torch_dtype = torch.bfloat16
        config.text_config.torch_dtype = torch.bfloat16
        with no_init_weights():
            model = GraphLlavaForConditionalGeneration(config)

    model.load_language_model()
    model.load_graph(graph_path)
    
    logger.info("Moving model to CUDA...")
    model.to(torch.device("cuda"))
    model.replace_mlp_with_moe()

    logger.info('Loading additional LLaVA weights...')
    non_lora_weights_path = os.path.join(model_path, 'non_lora_trainables.bin')

    if not os.path.exists(non_lora_weights_path):
        logger.warning(f"No Non-lora weights file found at: {non_lora_weights_path}")
        raise FileNotFoundError(f"Missing required file: {non_lora_weights_path}")

    non_lora_trainables = torch.load(
        non_lora_weights_path, map_location=torch.device("cuda"))
    
    logger.info(f"------ Non-lora trainables keys ------", on_rank0=True)
    for key in non_lora_trainables.keys():
        logger.info(f"{key}")
    logger.info(f"--------------------------------------", on_rank0=True)

    prefix_to_strip = "base_model.model."
    prefix_len = len(prefix_to_strip)
    non_lora_state_dict = {}
    for k, v in non_lora_trainables.items():
        if k.startswith(prefix_to_strip):
            non_lora_state_dict[k[prefix_len:]] = v
        else:
            logger.warning(f"Skipping non-lora key with unexpected format: {k}")

    model.load_state_dict(non_lora_state_dict, strict=False)

    logger.info('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)

    logger.info('Merging LoRA weights (on CUDA)...')
    model = model.merge_and_unload()

    logger.info("Moving final merged model back to CPU...")
    model.to(torch.device("cpu"))
    logger.info('Model is loaded and on CPU.')
    torch.cuda.empty_cache()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    return tokenizer, model, generation_config

def prepare_data(
    task_name: str, 
    tokenizer: PreTrainedTokenizer,
    data_path: str
):
    dataset = TASK_MAP[task_name](
        data_path=data_path,
        tokenizer=tokenizer,
        for_test=True
    )
    
    return [data_processing(raw, tokenizer) for raw in dataset]

def generate_one_sample(
    payload: dict,
    sampling_params: dict,
    model: GraphLlavaForConditionalGeneration,
    generation_config: GenerationConfig,
    tokenizer: PreTrainedTokenizer,
    device: str
):
    input_ids = payload["input_ids"].to(device)
    graphs = payload["graphs"]
    attention_mask = payload["attention_mask"].to(device)
    graphs = [x.to(device) if x is not None else None for x in graphs]
    output_ids = model.generate(
        input_ids,
        graphs=graphs,
        use_cache=True,
        attention_mask=attention_mask,
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        **sampling_params
    )
    
    result = {
        "prompt": payload["prompt"][0],
        "gt": payload["gt"][0],
        "pred": tokenizer.decode(output_ids[0][input_ids[0].shape[0]:])
    }
    return result


def single_gpu_inference(
    model: GraphLlavaForConditionalGeneration, 
    data: list[dict], 
    sampling_params: dict, 
    tokenizer: PreTrainedTokenizer, 
    generation_config: GenerationConfig, 
    gpu_id: int = 0
):
    """
    Runs inference on a single GPU.
    """
    device = f"cuda:{gpu_id}"
    print(f"--- Running single-GPU inference on {device} ---")
    
    model.to(device)
    model.eval()

    all_results = []
    
    def generate_and_store(payload, storage, do_print):
        result = generate_one_sample(
            payload, sampling_params, model, 
            generation_config, tokenizer, device
        )
        storage.append(result)
        if do_print:
            print(result)
    
    with torch.no_grad():
        do_print = True
        for item in tqdm(data, desc="Single-GPU Inference"):
            generate_and_store(item, all_results, do_print)
            do_print = False

    return all_results

def worker_fn(
    rank: int, 
    world_size, 
    model_path, 
    language_backbone, 
    graph_path, 
    task_name, 
    data_path, 
    prompt_version,
    sampling_params,
    temp_dir,
    seed=0
):
    """
    The main function for each parallel worker process.
    """
    print(f"[Rank {rank}] Worker process started.")
    setup_distributed(rank, world_size)
    # set_seed(seed)
    
    tokenizer, model, generation_config = build_model(
        model_path, language_backbone, graph_path
    )
    
    device = f"cuda:{rank}"
    model.to(device)
    model.eval()

    # Get this rank's chunk of data
    conversation_lib.default_conversation = conversation_lib.conv_templates[prompt_version]
    data = prepare_data(task_name, tokenizer, data_path)
    data_chunks = split_data(data, world_size)
    local_data = data_chunks[rank]
    
    print(f"[Rank {rank}] local data size {len(local_data)}")
    dist.barrier()
    
    local_results = []
    
    iterable = tqdm(
        local_data, 
        desc=f"Rank {rank} Inference", 
        position=rank,
        leave=True
    )
    
    def generate_and_store(payload, storage, do_print):
        result = generate_one_sample(
            payload, sampling_params, model, 
            generation_config, tokenizer, device
        )
        storage.append(result)
        if rank == 0 and do_print:
            print(result)
    
    with torch.no_grad():
        do_print = True
        for item in iterable:
            generate_and_store(item, local_results, do_print)
            do_print = False
            
    iterable.close()

    # Save this rank's results to a temporary file
    temp_file = os.path.join(temp_dir, f"results_rank_{rank}.pt")
    torch.save(local_results, temp_file)
    print(f"[Rank {rank}] Results saved to {temp_file}")

    dist.barrier()
    cleanup_distributed()
    print(f"[Rank {rank}] Worker process finished.")


def calculate_metrics(
    results: list[dict], 
    task_name: str, 
    metric_save_path: str = None, 
    eos_token: str = None, 
    tokenizer: PreTrainedTokenizer = None,
    drd2_scorer_path: str = None
):
    if task_name in (
        "forward", "reagent", "retrosynthesis", "solvent",
        "catalyst", "iupac", "textguidedmolgen"
    ):
        calculate_reaction_metrics(results, metric_save_path, eos_token=eos_token)
    elif task_name in (
        "weight", "tpsa", "logp"
    ):
        calculate_mae_with_text(results, metric_save_path, eos_token)
    elif task_name in ("molcap", "dqa", "experiment"):
        calculate_text_scores(results, metric_save_path, tokenizer, eos_token)
    elif task_name == "homolumo":
        calculate_homolumo_metrics(results, metric_save_path, eos_token)
    elif task_name in ("yield_BH", "yield_SM"):
        calculate_r2(results, metric_save_path, eos_token)
    elif task_name == "molediting":
        calculate_moledit_metrics(results, metric_save_path, drd2_scorer_path, eos_token)
    else:
        raise ValueError(f"Task name {task_name} is invalid!")
        

def main(args):
    assert torch.cuda.is_available(), "Inference supports GPU only!"
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise ValueError("Error: No GPUs available.")

    if args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs, but only {available_gpus} are available.")
        args.num_gpus = available_gpus
        
    set_seed(args.seed)
    model, generation_config = None, None
    if args.num_gpus == 1:
        print("Loading model onto CPU...")
        tokenizer, model, generation_config = build_model(
            args.model_path, args.language_backbone, args.graph_path
        )
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.prompt_version]
        data = prepare_data(args.task_name, tokenizer, args.data_path)
        print(f"Loaded data size {len(data)}")
    else:
        # In multi-GPU inference, we create model, dataset inside the worker,
        # to avoid shm overflow
        tokenizer = AutoTokenizer.from_pretrained(args.language_backbone)
        if tokenizer.pad_token is None:
            tokenizer.pad_token=tokenizer.eos_token

    all_results = []
    sampling_params = {
        "top_p": args.top_p,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": not args.deterministic
    }

    start_time = time.time()
    if args.num_gpus == 1:
        all_results = single_gpu_inference(
            model, data, sampling_params, tokenizer, 
            generation_config, gpu_id=0
        )
    else:
        print(f"--- Spawning {args.num_gpus} processes for parallel inference ---")
        world_size = args.num_gpus
        os.makedirs(args.temp_dir, exist_ok=True)
        
        worker_args = (
            world_size, args.model_path, args.language_backbone, args.graph_path, 
            args.task_name, args.data_path, args.prompt_version, sampling_params,
            args.temp_dir, args.seed
        )
        mp.spawn(
            worker_fn,
            args=worker_args,
            nprocs=world_size,
            join=True
        )
        
        print("All workers finished. Gathering results...")
        all_results_split = []
        for i in range(world_size):
            temp_file = os.path.join(args.temp_dir, f"results_rank_{i}.pt")
            try:
                chunk = torch.load(temp_file)
                all_results_split.append(chunk)
                os.remove(temp_file) # Clean up
            except FileNotFoundError:
                print(f"Error: Could not find temp file {temp_file}")
        
        all_results = [item for sublist in all_results_split for item in sublist]
        
        # Clean up temp directory
        try:
            os.rmdir(args.temp_dir)
        except OSError:
            pass # Directory might not be empty, which is fine.

    end_time = time.time()
    
    # --- Done ---
    print("\n--- Inference Complete ---")
    print(f"Total results gathered: {len(all_results)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    parent = os.path.dirname(args.answer_save_path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    save_json(all_results, args.answer_save_path)
    
    metric_save_path = None
    if args.save_metric:
        _filename = os.path.basename(args.answer_save_path)
        filename = os.path.splitext(_filename)[0]
        metric_filename = f"{filename}_metric.json"
        metric_save_path = os.path.join(parent, metric_filename)
    print(f"Calculating metrics and save to {metric_save_path}")
    calculate_metrics(
        all_results, args.task_name, metric_save_path, 
        LLAMA3_EOS, tokenizer, args.drd2_scorer_path
    )
    
    exit(0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference script for Omni-Mol")
    # Infra settings
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for inference. Defaults to 1."
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="cache/inference"
    )
    
    # Task information
    parser.add_argument(
        "--task_name",
        type=str,
        default="forward"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/test/forward.json"
    )
    
    # Model information
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/test/forward.json"
    )
    parser.add_argument(
        "--language_backbone",
        type=str,
        default="llama3"
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="moleculestm"
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="llama3"
    )
    parser.add_argument(
        "--drd2_scorer_path",
        type=str,
        required=True
    )
    
    # Sampling Params
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    
    # Save
    parser.add_argument(
        "--metric_save_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--answer_save_path",
        type=str,
        default="test/answer.json"
    )
    parser.add_argument(
        "--save_metric",
        action="store_true",
        default=False
    )
    
    args = parser.parse_args()
    
    main(args)
