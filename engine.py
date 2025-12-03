from transformers import Trainer
import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
from deepspeed.moe.utils import is_moe_param, split_params_into_different_moe_groups_for_optimizer
from transformers.utils import is_peft_available, is_datasets_available
import importlib
from packaging import version
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import numpy as np
from utils import maybe_zero_3
from transformers.trainer_utils import EvalLoopOutput
from typing import Optional, List
from loggers import WrappedLogger

logger = WrappedLogger(__name__)

if is_peft_available():
    from peft import PeftModel
    
if is_datasets_available():
    import datasets

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


class MoETrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is not None:
            return self.optimizer

        # Separate decay parameters, excluding bias and layernorms
        decay_parameters = [
            n for n in get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            if "bias" not in n
        ]
        
        # Group parameters for weight decay and no decay
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "name": "decay_parameters"
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay_parameters"
            },
        ]
        
        # Log MoE parameters
        for name, param in opt_model.named_parameters():
            if is_moe_param(param):
                logger.info(f"Detected MoE parameters: {name}", on_rank0=True)

        if self.args.moe_enable:
            logger.info(f"Splitting params for MoE...", on_rank0=True)
            optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(
                optimizer_grouped_parameters
            )

        # Get optimizer class and arguments
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        outputs = model(**inputs)
        # log auxiliary loss
        # NOTE: not compatible with tuple data type
        aux_log = None
        if getattr(outputs, "moe_loss", None):
            aux_log = {
                "model_loss": self.accelerator.reduce(outputs.model_loss,reduction="mean").item()
            }
            
        if getattr(outputs, "moe_loss", None) is not None:
            aux_log.update({"moe_loss": self.accelerator.reduce(outputs.moe_loss,reduction="mean").item()})
            aux_log.update({"router_coeff": outputs.router_aux_coeff})
            
        if aux_log is not None:
            self.log(aux_log)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
