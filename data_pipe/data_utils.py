import torch
from torch_geometric.data import Batch, Data
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple, List

IGNORE_INDEX = -100

@dataclass       
class GraphDatasetCollator(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, task_ids = self._extract_tensors(instances, ("input_ids", "labels", "this_task_ids"))
        
        input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        labels = self._pad_sequence(labels, IGNORE_INDEX)
        task_ids = torch.cat(task_ids, dim=0)
        batch = {
            'input_ids': input_ids[:, :self.tokenizer.model_max_length],
            'labels': labels[:, :self.tokenizer.model_max_length],
            'attention_mask': input_ids[:, :self.tokenizer.model_max_length].ne(self.tokenizer.pad_token_id),
            'this_task_ids': task_ids  # shape of B
        }
        if 'graphs' in instances[0]:
            graph_batch = []
            for instance in instances:
                if instance["graphs"] is not None:
                    graph_batch.append(self._convert_dict_to_Data(instance["graphs"]))
                else:
                    graph_batch.append(None)
            batch["graphs"] = graph_batch
            # batch['graphs'] = Batch.from_data_list(
            #     [self._convert_dict_to_Data(instance["graphs"]) for instance in instances]
            # )
        return batch
    
    def _extract_tensors(self, instances, keys: Tuple[str, str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return tuple([instance[key] for instance in instances] for key in keys)

    def _pad_sequence(self, sequence: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True, padding_value=padding_value)

    def _convert_dict_to_Data(self, data_dict: Dict) -> Data:
        if getattr(data_dict, "num_part", None) is not None: # which means we are using himol
            return Data(
            x=torch.asarray(data_dict.x),
            edge_index=torch.asarray(data_dict.edge_index),
            edge_attr=torch.asarray(data_dict.edge_attr),
            num_part=data_dict.num_part
            )
            
        return Data(
            x=torch.asarray(data_dict['node_feat']),
            edge_attr=torch.asarray(data_dict['edge_feat']),
            edge_index=torch.asarray(data_dict['edge_index']),
        )