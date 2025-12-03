"""
@ 2025 Omni-Mol Project

Main training script of Omni-Mol
WARNING: This is a preview version!
"""
from loggers import WrappedLogger
from arguments import ModelArguments, DataArguments, TrainingArguments
import transformers
from transformers import TrainerCallback
from transformers import TrainerControl
from dataclasses import asdict
import json
import os
import datetime
import engine
from utils import model_profiler, seperate_save_lora
from model import create_model
import pathlib
from typing import Any, Optional
from utils import save_json
import torch
from torch.utils.data import Subset, ConcatDataset
from constants import TASK_MAP, MAP2FILE
from data_pipe.data_utils import GraphDatasetCollator
from data_pipe import conversation_lib


logger = WrappedLogger(__name__)

def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", -1)) in [0, -1]

def save_json_on_main(data: Any, path: str) -> None:
    if is_main_process():
        save_json(data, path)
        
def get_last_ckpt(output_dir: str) -> Optional[bool]:
    if list(pathlib.Path(output_dir).glob("checkpoint-*")):
        return True
    return None

def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args, training_args

def create_mixed_dataset(datasets_dict: dict, config: dict, data_path, tokenizer, file_names, shuffle=True, add_all=False):
    subsets = []
    logger.info(f"{'Task Name':<20} | {'Original Size':<15} | {'Weight':<10} | {'Final Size':<10}", on_rank0=True)
    logger.info("-" * 65, on_rank0=True)

    if add_all:
        task_iterator = [(name, 1.0) for name in datasets_dict.keys()]
    else:
        task_iterator = config.items()

    for task_name, weight in task_iterator:
        if task_name not in datasets_dict:
            raise ValueError(f"Task '{task_name}' found in config but not in datasets_dict.")
        
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Weight for '{task_name}' must be between 0 and 1. Got {weight}.")
        
        file_name = file_names[task_name]
        dataset = datasets_dict[task_name](
            data_path=os.path.join(data_path, f"{file_name}.json"),
            tokenizer=tokenizer,
            for_test=False
        )
        total_count = len(dataset)
        target_count = int(total_count * weight)

        # Handle the case where weight results in 0 samples
        if target_count == 0:
            logger.info(f"{task_name:<20} | {total_count:<15} | {weight:<10.2f} | 0 (Skipped)", on_rank0=True)
            continue

        if shuffle:
            # Randomly select indices
            indices = torch.randperm(total_count)[:target_count].tolist()
        else:
            # Take the first N indices
            indices = list(range(target_count))

        # Create a Subset for this task
        task_subset = Subset(dataset, indices)
        subsets.append(task_subset)

        logger.info(f"{task_name:<20} | {total_count:<15} | {weight:<10.2f} | {target_count:<10}", on_rank0=True)

    # Combine all subsets into one dataset
    mixed_dataset = ConcatDataset(subsets)
    logger.info("-" * 65, on_rank0=True)
    logger.info(f"Total Mixed Dataset Size: {len(mixed_dataset)}", on_rank0=True)
    
    return mixed_dataset


def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    # seed everything (Optional)
    transformers.set_seed(0)
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    # Dump args
    args = {
        "Model Args": asdict(model_args), 
        "Data Args": asdict(data_args), 
        "Training Args": asdict(training_args)
        }
    if int(os.environ.get("LOCAL_RANK",-1)) in [0, -1]:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        save_json(args, os.path.join(training_args.output_dir, "args.json"))
    
    # Create model, tokenizerç
    tokenizer, model = create_model(model_args, training_args)
    if tokenizer.pad_token is None and model_args.version == "llama3":
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        
    logger.info(f"Using conversation template of {model_args.version}", on_rank0=True)
    logger.info(f"Conversation template: {conversation_lib.default_conversation}", on_rank0=True)
    # create dataset
    TASK_MAP.pop("yield_BH")
    cls = TASK_MAP.pop("yield_SM")
    TASK_MAP["yield"] = cls
    train_dataset = create_mixed_dataset(
        datasets_dict=TASK_MAP,
        config=None,
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        file_names=MAP2FILE,
        shuffle=False, 
        add_all=True
    )
    data_collator = GraphDatasetCollator(tokenizer=tokenizer)
    
    model_profiler(model, training_args.output_dir)
    
    # callback function for model saving
    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # get saving dir from args
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            seperate_save_lora(args, checkpoint_dir, model)

            
    class TruncateCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if args.stop_epoch is not None:
                if state.epoch > args.stop_epoch:
                    return TrainerControl(should_epoch_stop=True)
    
    # train
    trainer = engine.MoETrainer(
        model=model,
        args=training_args,
        callbacks=[SaveCallback(), TruncateCallback()],
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=None
    )

    resume_ckpt = get_last_ckpt(training_args.output_dir)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    

if __name__ == "__main__":
    logger.info(f"Training script for Omni-Mol", on_rank0=True)
    logger.info(f"Time: \033[34m{datetime.datetime.now()}\033[0m", on_rank0=True)
    model_args, data_args, training_args = parse_args()
    main(model_args, data_args, training_args)
