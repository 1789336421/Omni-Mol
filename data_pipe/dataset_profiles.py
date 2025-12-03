"""
@ 2025 Omni-Mol Project

Tasks and the pre-processing routine
"""
import json
import selfies
import torch
import pandas as pd
import random
import re

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional
from data_pipe import conversation_lib
from data_pipe.chat_template import apply_chat_template, tokenizer_image_token
from data_pipe.mol_utils import smiles2graph
from loggers import WrappedLogger


logger = WrappedLogger(__name__)

def check_output(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: PreTrainedTokenizer
):
    ids = list(input_ids.detach().numpy())
    logger.info(f"ids: {ids}")
    logger.info(f"labels {labels}")
    if -200 in ids:
        ids.remove(-200)
    logger.info([tokenizer.decode(ids)])
    

def apply_prompt(message):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    
    return prompt


@dataclass
class SampleInfo:
    instruction: str
    smiles_for_graph: str
    target_text: str
    
class BaseGraphTaskDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        tokenizer: PreTrainedTokenizer, 
        for_test: bool, 
        task_name: str,
        data_file: list[dict] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.for_test = for_test
        self.task_name = task_name
        
        # 1. Data Loading
        if data_file is not None:
            # Direct apply of data file
            self.list_data_dict = data_file
        else:
            if data_path.endswith('.csv'):
                self.list_data_dict = pd.read_csv(data_path).to_dict('records')
            else:
                with open(data_path, "rb") as f:
                    self.list_data_dict = json.load(f)

        # 2. Filtering
        if not data_path.endswith('.csv'):
            # which means we are in Apache style json data
            assert "metadata" in self.list_data_dict[0], f"Missing required filed metadata."
            target_split = 'test' if for_test else 'train'
            self.list_data_dict = [
                raw for raw in self.list_data_dict 
                if raw['metadata']['split'] == target_split
            ]

        logger.info(f"Task {task_name}: {len(self)} samples loaded.", on_rank0=True)

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def selfies2smiles(self, selfies_str: str) -> Optional[str]:
        try:
            return selfies.decoder(selfies_str)
        except:
            return None

    # Must be implemented in sub task classes
    def parse_sample(self, raw_data) -> SampleInfo:
        raise NotImplementedError("Subclasses must implement parse_sample")
    
    def adjust_messages_if_needed(self, data_dict, messages):
        """Optional hook for subclasses to modify messages before tokenization"""
        return data_dict

    def __getitem__(self, i) -> dict:
        raw = self.list_data_dict[i]
        
        # 1. Pre-hook for sample parsing
        info = self.parse_sample(raw)
        
        # 2. Common Pre-processing
        full_instruction = f"<image>\n{info.instruction}"
        graph = smiles2graph(info.smiles_for_graph)
        
        if graph is None:
            raise ValueError(f"Could not convert SMILES to graph: {info.smiles_for_graph}")

        # 3. Test Mode Routine
        if self.for_test:
            prompt = apply_prompt(full_instruction)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
            return {
                "input_ids": input_ids,
                "graphs": graph,
                "gt": info.target_text,
                "prompt": prompt,
            }

        # 4. Training Mode Routine
        messages = [[
            {"from": "human", "value": full_instruction},
            {"from": "gpt", "value": info.target_text}
        ]]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=True)
        
        # Post-Hook
        data_dict = self.adjust_messages_if_needed(data_dict, messages)
        
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]
        
        assert -200 in input_ids, "Input IDs missing expected <image> token"

        return {
            "input_ids": input_ids,
            "labels": labels,
            "graphs": graph,
        }

class ForwardPredDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test, 
            task_name="ForwardPred", **kwargs
        )

    def parse_sample(self, raw) -> SampleInfo:
        inputs = raw['input'].split('.')
        reactant_smiles = self.selfies2smiles(inputs[0])
        
        instruction = f"{raw['instruction']} {raw['input']}"
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=reactant_smiles,
            target_text=raw['output']
        )

class ReagentPredDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="ReagentPred", **kwargs
        )

    def parse_sample(self, raw) -> SampleInfo:
        reactant_selfies, _ = raw['input'].split(">>")
        reactant_smiles = self.selfies2smiles(reactant_selfies)
        
        instruction = f"{raw['instruction']} The reaction is {raw['input']}"

        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=reactant_smiles,
            target_text=raw['output']
        )
        
class RetrosynDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Retrosynthesis", **kwargs
        )
        
    def parse_sample(self, raw) -> SampleInfo:
        product_smiles = self.selfies2smiles(raw['input'])
        instruction = f"{raw['instruction']} The product is: {raw['input']}"
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=product_smiles,
            target_text=raw['output']
        )
        
class PropertyPredDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="HOMOLUMO Prediction", **kwargs
        )
        
    def parse_sample(self, raw):
        molecule_smiles = self.selfies2smiles(raw['input'])
        instruction = f"{raw['instruction']} The compound SELFIES sequence is: {raw['input']}"
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=molecule_smiles,
            target_text=str(raw['output'])
        )
        
class MolcapDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        self.question_pool = [
            'Could you give me a brief overview of this molecule?',
            'Could you provide a description of this molecule?',
            'Describe this molecule.',
            'Please give me some details about this molecule.',
            'Provide a brief overview of this molecule.',
            'Provide a description of this molecule.',
            'What can you tell me about this molecule?'
        ]
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Molcap", **kwargs
        )
        
    def parse_sample(self, raw):
        prompt = random.choice(self.question_pool)
        instruction = f"{prompt} The compound sequence is: {raw['input']}"
        molecule_smiles = self.selfies2smiles(raw['input'])
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=molecule_smiles,
            target_text=raw['output']
        )
        
    def adjust_messages_if_needed(self, data_dict, messages):
        """Overlong filtering"""
        if not len(data_dict['input_ids'][0]) > self.tokenizer.model_max_length:
            return data_dict
        
        logger.warning(f"input too long {len(data_dict['input_ids'][0])}, selfies dropped.")
        instruction = random.choice(self.question_pool)
        instruction = "<image>\n" + instruction
        messages[0][0]["value"] = instruction
        data_dict = apply_chat_template(messages, self.tokenizer, has_image=True)
        logger.warning(f"Adjusted length {len(data_dict['input_ids'][0])}")

        return data_dict
    
class CatalystPredDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Catalyst Prediction", **kwargs
        )
        
    def parse_sample(self, raw):
        reactant, _ = raw['input'].split(">>")
        reactant_smiles = self.selfies2smiles(reactant)
        instruction = f"{raw['instruction']} The reaction is {raw['input']}."
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=reactant_smiles,
            target_text=raw['output']
        )
        
class SolventPredDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Solvent Prediction", **kwargs
        )

    def parse_sample(self, raw):
        reactant, _ = raw['input'].split(">>")
        reactant_smiles = self.selfies2smiles(reactant)
        instruction = f"{raw['instruction']} The reaction is {raw['input']}."
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=reactant_smiles,
            target_text=raw['output']
        )
        
class YieldRegressionDataset(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Yield Regression", **kwargs
        )
        
    def parse_sample(self, raw):
        reactant, _ = raw['input'].split(">>")
        reactant_smiles = self.selfies2smiles(reactant)
        instruction = f"{raw['instruction']} The reaction is {raw['input']}."
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=reactant_smiles,
            target_text=str(raw['output'])
        )
        
class ExpProcedurePrediction(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Experiment Procedure Prediction", **kwargs
        )
        
    def parse_sample(self, raw):
        extracted_molecules = raw.get("extracted_molecules", {})
        placeholder_to_smiles = {placeholder: smi for smi, placeholder in extracted_molecules.items()}

        placeholders = re.findall(r"\$\d+\$", raw["input"])

        smiles_list = []
        for ph in placeholders:
            if ph in placeholder_to_smiles:
                smiles_list.append(placeholder_to_smiles[ph])

        smiles = ".".join(smiles_list)
        instruction = f"{raw['instruction']}{raw['input']}. The Action Sequence: "

        assert smiles is not None, f"Found invalid data {raw}"
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=smiles,
            target_text=raw['output']
        )
        
class LogPPrediction(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="LogP", **kwargs
        )
        
    def parse_sample(self, raw):
        instruction = f"{raw['instruction']} The molecule SELFIES sequence is: {raw['input']}"
        molecule_smiles = self.selfies2smiles(raw['input'])
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=molecule_smiles,
            target_text=raw['output']
        )
        
class DescriptionQA(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Description QA", **kwargs
        )
        
    def parse_sample(self, raw):
        instruction = f"{raw['instruction']} The compound SELFIES sequence is: {raw['input']}"
        to_graph = self.selfies2smiles(raw['input'].split(".")[0])
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=to_graph,
            target_text=raw['output']
        )
        
class WeightPrediction(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Weight", **kwargs
        )
        
    def parse_sample(self, raw):
        instruction = f"{raw['instruction']} The molecule SELFIES sequence is: {raw['input']}"
        molecule_smiles = self.selfies2smiles(raw['input'])
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=molecule_smiles,
            target_text=raw['output']
        )
        
class TPSAPrediction(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="TPSA", **kwargs
        )
        
    def parse_sample(self, raw):
        instruction = f"{raw['instruction']} The compound SELFIES sequence is: {raw['input']}"
        to_graph = self.selfies2smiles(raw['input'].split(".")[0])
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=to_graph,
            target_text=raw['output']
        )
        
class IUPAC(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="IUPAC", **kwargs
        )
        
    # Override for pure text tasks
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = f"{raw['instruction']} The IUPAC name is: {raw['input']}"
        graph = None
    
        if self.for_test:
            prompt = apply_prompt(instruction)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
            return {
                "input_ids": input_ids,
                "graphs": graph,
                "gt": raw['output'],
                "prompt": prompt,
            }
        
        messages = [[
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": raw['output']}
        ]]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "graphs": graph,
        }
        
class TextGuidedMolGen(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Text Guided Molecule Generation", **kwargs
        )
        
    # Override for pure text tasks
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = f"{raw['instruction']} The description is: {raw['input']}"
        graph = None
    
        if self.for_test:
            prompt = apply_prompt(instruction)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
            return {
                "input_ids": input_ids,
                "graphs": graph,
                "gt": raw['output'],
                "prompt": prompt,
            }
        
        messages = [[
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": raw['output']}
        ]]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "graphs": graph,
        }
        
class MolEditing(BaseGraphTaskDataset):
    def __init__(self, data_path, tokenizer, for_test, **kwargs):
        super().__init__(
            data_path, tokenizer, for_test,
            task_name="Molecule Editing", **kwargs
        )
    
    def parse_sample(self, raw):
        instruction = f"{raw['instruction']} The compound SELFIES sequence is: {raw['input']}"
        to_graph = self.selfies2smiles(raw['input'])
        
        return SampleInfo(
            instruction=instruction,
            smiles_for_graph=to_graph,
            target_text=raw['output']
        )
