import json
import selfies
import torch
import pandas as pd
import random
import re

from transformers import PreTrainedTokenizer
from data_pipe.chat_template import apply_chat_template
from data_pipe.chat_template import tokenizer_image_token
from torch.utils.data import Dataset
from data_pipe.mol_utils import smiles2graph
from loggers import WrappedLogger
from data_pipe import conversation_lib


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


class MetaGraphDataset(Dataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        task_name: str,
        data_file: list[dict] = None
    ) -> None:
        super().__init__()
        if data_file is None:
            with open(data_path, "rb") as f:
                self.list_data_dict = json.load(f)
                f.close()
        else:
            self.list_data_dict = data_file

        self.tokenizer = tokenizer
        self.task_name = task_name
        logger.info(f"Task \033[34m{task_name}\033[0m\t Total number of samples: {self.__len__()}", on_rank0=True)
        if for_test:
            self.filter_for_test()
            logger.info(f"Filtered {self.__len__()} for test", on_rank0=True)
        else:
            self.filter_for_training()
            logger.info(f"Filtered {self.__len__()} for training", on_rank0=True)
        
    def selfies2smiles(self, selfies_str: str) -> str | None:
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
            
        return smiles_str


    def filter_for_training(self) -> None:
        self.list_data_dict = [raw for raw in self.list_data_dict if raw['metadata']['split'] == 'train']
    
    def filter_for_test(self) -> None:
        self.list_data_dict =  [raw for raw in self.list_data_dict if raw['metadata']['split'] == 'test']
        
    def _yield_prompt(self, instruction, graphs, gt):
        prompt = apply_prompt(instruction)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
    
        data = {
            "input_ids": input_ids,
            "graphs": graphs,
            "gt": gt,
            "prompt": prompt,
            "this_task_ids": torch.LongTensor([0])
        }
        
        return data
        
    def __len__(self) -> int:
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        pass
    
    
class PretrainMolDataset(Dataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        data_file: list[dict] = None,
        **kwargs
    ) -> None:
        super().__init__()
        if data_file is None:
            list_data_dict = pd.read_csv(data_path)
            self.list_data_dict = list_data_dict
        else:
            self.list_data_dict = data_file
            
        self.tokenizer = tokenizer
        
        logger.info(f"Total number of samples: {self.__len__()}", on_rank0=True)
        print("====Pretrain Molecule Description Dataset====")
        
        self.question_pool = [
            'Could you give me a brief overview of this molecule?',
            'Could you provide a description of this molecule?',
            'Describe this molecule.',
            'Please give me some details about this molecule.',
            'Provide a brief overview of this molecule.',
            'Provide a description of this molecule.',
            'What can you tell me about this molecule?'
        ]
        
        
    def __len__(self):
        return len(self.list_data_dict)
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        smiles, description = self.list_data_dict["SMILES"][i], self.list_data_dict["Description"][i]
        
        instruction = random.choice(self.question_pool)
        instruction = "<image>\n" + instruction
        
        message = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": description}
            ]
        ]
        
        graph_for_molecule = smiles2graph(smiles)
        assert graph_for_molecule is not None, f"Found molecule that cannot be converted to graph: {smiles}"
        
        data_dict = apply_chat_template(message, self.tokenizer, graph_for_molecule is not None)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        
        data_dict['graphs'] = graph_for_molecule
        assert -200 in data_dict["input_ids"], "Input IDs missing expected <image> token"

        return data_dict


class ForwardPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
        ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Forward Prediction==",
            data_file
        )
        self.for_test = for_test
                
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # 1. Get sample
        raw = self.list_data_dict[i]

        # 2. Get instruction, input selfies, output selfies
        instruction = raw['instruction']
        inputs, output_selfies = raw['input'].split('.'), raw['output']
        
        # 3. Convert to Graph
        reactant_smiles = self.selfies2smiles(inputs[0])
        graph_for_first_reactant = smiles2graph(reactant_smiles)

        # 4. Add SELFIES
        instruction += " " + raw['input']
        instruction = "<image>\n" + instruction
        
        # test routine
        if self.for_test:
            return self._yield_prompt(instruction, graph_for_first_reactant, output_selfies)
        
        # 5. Prepare conversations
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        # Tokenization
        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph_for_first_reactant is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        
        assert graph_for_first_reactant is not None, f"Cannot convert {inputs[0]} to graph"
        data_dict['graphs'] = graph_for_first_reactant
        assert -200 in data_dict["input_ids"], "Input IDs missing expected <image> token"
        data_dict["this_task_ids"] = torch.LongTensor([0])

        return data_dict
    
    
class ReagentPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Reagent Prediction==",
            data_file
        )
        
        self.for_test = for_test
        
    @staticmethod
    def construct_instruct_question(product:str):
        """
        Construct instruct question for each graph
        """
        question_pools = [
            'Can you suggest some possible reagents that could have been used in the following chemical reaction?',
            'Give some possible reagents that could have been used in the following chemical reaction.',
            'Please propose potential reagents that might have been utilized in the provided chemical reaction.',
            'Please provide possible reagents based on the following chemical reaction.',
        ]
        question = random.choice(question_pools)
        question += f"\nThe product is {product}"
        return question    

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        instruction = raw['instruction'] + f" The reaction is {input}"

        instruction = "<image>\n" + instruction
        
        graph=smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])


        assert graph is not None, f"Cannot convert {reactant} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([1])
            
        return data_dict
    
    
class RetrosynDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Retrosynthesis==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        instruction += f" The product is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, output_selfies = raw['input'], raw['output']
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(input_selfies)
        
        graph=smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input_selfies} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([2])
        
        return data_dict
    
    
class PropertyPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==HOMO LUMO==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        instruction += f" The compound SELFIES sequence is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, target = raw['input'], str(raw['output'])
        graph=smiles2graph(self.selfies2smiles(input_selfies))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, target)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": target}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input_selfies} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([3])
        
        return data_dict
    
class MolcapDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Molcap==",
            data_file,
            **kargs
        )
        self.question_pool = [
            'Could you give me a brief overview of this molecule?',
            'Could you provide a description of this molecule?',
            'Describe this molecule.',
            'Please give me some details about this molecule.',
            'Provide a brief overview of this molecule.',
            'Provide a description of this molecule.',
            'What can you tell me about this molecule?'
        ]
        self.for_test = for_test
        
    def maybe_drop_selfies(
        self,
        data_dict,
        messages,
        has_image
    ):
        if len(data_dict['input_ids']) > self.tokenizer.model_max_length:
            logger.warning(f"input too long {len(data_dict['input_ids'])}, selfies dropped.")
            instruction = random.choice(self.question_pool)
            instruction = "<image>\n" + instruction
            messages[0][0]["value"] = instruction
            data_dict = apply_chat_template(messages, self.tokenizer, has_image=has_image)
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
            logger.warning(f"Adjusted length {len(data_dict['input_ids'])}")
            
        return data_dict
        
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = random.choice(self.question_pool)
        input = raw['input']
        output = raw['output']
        
        instruction += f" The compound sequence is: {input}"

        instruction = "<image>\n" + instruction
        
        graph = smiles2graph(self.selfies2smiles(input))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)
        
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        
        data_dict = self.maybe_drop_selfies(data_dict, messages, has_image=(graph is not None))
        
        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([4])

        return data_dict
    
    
class CatalystPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Catalyst Prediction dataset==",
            data_file
        )
        self.for_test = for_test

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        # insert product to the instruction end
        instruction = raw['instruction'] + f" The reaction is {input}."

        instruction = "<image>\n" + instruction
        graph = smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([5])

        return data_dict


class SolventPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Solvent Prediction dataset==",
            data_file
        )
        self.for_test = for_test

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        # insert product to the instruction end
        instruction = raw['instruction'] + f" The reaction is {input}."

        instruction = "<image>\n" + instruction

        graph = smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([6])

        return data_dict


class YieldRegressionDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Yield Regression dataset==",
            data_file
        )
        self.for_test = for_test

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        instruction = raw['instruction'] + f" The reaction is {input}."

        instruction = "<image>\n" + instruction
        graph = smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": str(output_selfies)}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([7])

        return data_dict
    
class ExpProcedurePrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Experimental Procedure Prediction dataset==",
            data_file
        )
        self.for_test = for_test

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]

        extracted_molecules = raw.get("extracted_molecules", {})

        placeholder_to_smiles = {placeholder: smi for smi, placeholder in extracted_molecules.items()}

        placeholders = re.findall(r"\$\d+\$", raw["input"])

        smiles_list = []
        for ph in placeholders:
            if ph in placeholder_to_smiles:
                smiles_list.append(placeholder_to_smiles[ph])

        smiles = ".".join(smiles_list)

        input, output_selfies = raw['input'], raw['output']
        instruction = raw['instruction'] + f"{input}. "
        instruction += "The Action Sequence: "
        instruction = "<image>\n" + instruction

        assert smiles is not None, f"Found invalid data {raw}"
        graph = smiles2graph(smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([8])

        return data_dict
    
    
class SCFPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==SCF Prediction==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        output = raw["output"]
        
        instruction += f" The molecule SELFIES sequence is: {mol}"

        instruction = "<image>\n" + instruction
        
        graph = smiles2graph(self.selfies2smiles(mol))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([9])

        return data_dict
    
    
class LogPPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==LogP Prediction==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        output = raw["output"]
        
        instruction += f" The molecule SELFIES sequence is: {mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(mol))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([10])

        return data_dict
    

class DescriptionQA(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Description QA==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        to_graph = mol.split(".")[0]
        output = raw["output"]
        
        instruction += f" The compound SELFIES sequence is: {mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(to_graph))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([11])

        return data_dict
    
class WeightPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Weight Prediction==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        output = raw["output"]
    
        instruction += f" The molecule SELFIES sequence is: {mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(mol))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([12])

        return data_dict
    
class TPSAPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Topological Polar Surface Area==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        to_graph = mol.split(".")[0]
        output = raw["output"]
        
        instruction += f" The compound SELFIES sequence is: {mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(to_graph))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([13])

        return data_dict
    
    
class ComlexityPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Complexity==",
            data_file
        )
        self.for_test = for_test
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        to_graph = mol.split(".")[0]
        output = raw["output"]
        
        instruction += f" The compound SELFIES sequence is: {mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(to_graph))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([14])

        return data_dict
    
class IUPAC(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==IUPAC==",
            data_file
        )

        self.for_test = for_test

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
    
        iupac, output_selfies = raw['input'], raw['output']
        instruction += f" The IUPAC name is: {iupac}" 
        graph = None
        # graph=smiles2graph(self.selfies2smiles(input_selfies))
    
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)
        
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])

        data_dict['graphs'] = graph

        data_dict["this_task_ids"] = torch.LongTensor([15])
    
        return data_dict
        
class TextGuidedMolGen(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Text Guided Mol Gen==",
            data_file
        )

        self.for_test = for_test

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
    
        desc, output_selfies = raw['input'], raw['output']
        instruction += f" The description is: {desc}" 
        graph = None
    
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)
        
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])

        # assert graph is not None, "Cannot convert to graph"
        data_dict['graphs'] = graph
        assert -200 not in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([16])
    
        return data_dict
        
        
class MolEditing(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        data_file: list[dict] = None
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==molecular editing==",
            data_file
        )

        self.for_test = for_test

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        instruction += f" The compound SELFIES sequence is: {raw['input']}"

        instruction = "<image>\n" + instruction
    
        input_selfies, output_selfies = raw['input'], raw['output']
        # instruction += f" The description is: {desc}" 
        # graph = None
        graph=smiles2graph(self.selfies2smiles(input_selfies))
    
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)
        
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input_selfies} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([17])
    
        return data_dict
