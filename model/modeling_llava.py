"""
@ 2025 Omni-Mol Project

TODO: Refactor this to support newer transformers version
"""
import deepspeed
import torch

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache
from model.configs import GraphLlavaConfig
from model.modeling_projector import NAME2PROJ
from model.modeling_moleculestm import GraphTower
from model.modeling_moe import DeepseekV2MoE
from model.modeling_llama import MoELlamaDecoderLayerForward, MoELlamaModelForward
from constants import MODEL_CLS_MAP
from torch import nn
from torch.nn import CrossEntropyLoss
from typing_extensions import Optional, Union, Tuple, List
from dataclasses import dataclass

from loggers import WrappedLogger
logger = WrappedLogger(__name__)

IGNORE_INDEX = -100

FORWARD_MAP = {
    "llama": [MoELlamaDecoderLayerForward, MoELlamaModelForward],
}

@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    model_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    router_aux_coeff: Optional[float] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.Tensor] = None


class LlavaPreTrainedModel(PreTrainedModel):
    config_class = GraphLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa
    

class GraphLlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
    def __init__(self, config: GraphLlavaConfig):
        """The class for the Graph LLaVA model

        Args:
            config (GraphLlavaConfig): Config of GraphLLaVA
        """
        super().__init__(config)
        # GRAPH TOWER ==============
        self.graph_tower = GraphTower(config.graph_config)
        
        # PROJECTOR ===================
        self.mm_projector = NAME2PROJ[config.projector_type](config.graph_config.hidden_size, config.text_config.hidden_size)
        
        # LLM =================================
        self.vocab_size = config.text_config.vocab_size
        lm_cls = MODEL_CLS_MAP[config.language_backbone_name]
        logger.info(f"Equipped with {config.language_backbone_name}", on_rank0=True)
        self.language_model = lm_cls._from_config(
            config.text_config,
            attn_implementation=config.text_config._attn_implementation, 
            torch_dtype=self.config.text_config.torch_dtype
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.config.hidden_size = self.config.text_config.hidden_size
        
        # self.apply(self._init_weight)
        
    def load_graph(self, path2graph: str) -> None:
        """Load graph ckpt to the model's graph tower

        Args:
            path2graph (str): path to the graph ckpt, e.g.(himol_encoder.pth)
        """
        logger.info(f"Loading graph ckpt from {path2graph}", on_rank0=True)
        self.graph_tower.gnn.load_weight(path2graph)
        
    def load_projector(self, path2proj: str):
        """Load projector weight

        Args:
            path2proj (str): path to the projector weight
        """
        logger.info(f"Lodaing projector from {path2proj}", on_rank0=True)
        state_dict = torch.load(path2proj, weights_only=True)
        logger.info(f"Projector State Dict: {state_dict.keys()}", on_rank0=True)
        state_dict = {k.split("mm_projector.")[1]: v for k, v in state_dict.items()}
        self.mm_projector.load_state_dict(state_dict)
        
    def load_language_model(self):
        """Load LLM, the LLM type & path is specified in the text config
        """
        logger.info(f"Loading LLM ckpt from {self.config.text_config._name_or_path}", on_rank0=True)
        self.language_model = self.language_model.from_pretrained(
            self.config.text_config._name_or_path,
            config=self.config.text_config,
            torch_dtype=self.config.text_config.torch_dtype,
            attn_implementation=self.config.text_config._attn_implementation
            )
        
    def rand_init_projector(self):
        # Let's create the projector again, this can perfectly simulate
        # the enviroment without _no_init_weight context manager
        self.mm_projector = NAME2PROJ[self.config.projector_type](self.config.graph_config.hidden_size, self.config.text_config.hidden_size)
        
    def replace_mlp_with_moe(self):
        # Get configuration values
        num_layers = self.config.text_config.num_hidden_layers
        moe_layers_idx = self.config.moe_config.moe_layers_idx

        # Set MoE layers if not already defined
        if moe_layers_idx is None:
            moe_layers_idx = self._get_moe_layers_by_mode(num_layers)

        # Validate MoE layers
        self._validate_moe_layers(num_layers, moe_layers_idx)

        # Set experts per layer
        self._set_experts_for_layers(moe_layers_idx)

        # Log MoE configuration
        self._log_moe_configuration(num_layers, moe_layers_idx)

        # Replace MLP with MoE for the selected layers
        self._replace_mlp_with_moe_for_layers(moe_layers_idx)

        # Update the forward functions for the model
        self._update_forward_functions()

    def _get_moe_layers_by_mode(self, num_layers):
        """
        Determine the MoE layers based on the configuration mode.
        """
        mode = self.config.moe_config.moe_mode
        if mode == "first_half":
            return list(range(0, num_layers // 2))
        elif mode == "second_half":
            return list(range(num_layers // 2, num_layers))
        elif mode == "sparse":
            return list(range(num_layers))[::4]
        elif mode == "dense":
            return list(range(num_layers))
        elif mode == "second_quarter":
            return list(range(num_layers - (num_layers // 4), num_layers))
        else:
            raise NotImplementedError(f"Unsupported moe_mode: {mode}")

    def _validate_moe_layers(self, num_layers, moe_layers_idx):
        """
        Validate the MoE layer indices.
        """
        assert len(moe_layers_idx) <= num_layers
        assert max(moe_layers_idx) < num_layers
        assert min(moe_layers_idx) >= 0

    def _set_experts_for_layers(self, moe_layers_idx):
        """
        Set the number of experts for each MoE layer.
        """
        if isinstance(self.config.moe_config.num_experts, int):
            self.config.moe_config.num_experts = [self.config.moe_config.num_experts] * len(moe_layers_idx)

        assert len(self.config.moe_config.num_experts) == len(moe_layers_idx)

    def _replace_mlp_with_moe_for_layers(self, moe_layers_idx):
        """
        Replace MLP with MoE in the selected layers.
        """
        for num_experts, layer_num in zip(self.config.moe_config.num_experts, moe_layers_idx):
            expert = self.language_model.model.layers[layer_num].mlp
            pretrained_state_dict = expert.state_dict()

            if self.config.moe_config.moe_class == "deepspeed":
                self.language_model.model.layers[layer_num].mlp = deepspeed.moe.layer.MoE(
                    self.config.text_config.hidden_size,
                    expert=expert,
                    num_experts=num_experts,
                    ep_size=self.config.moe_config.ep_size,
                    k=self.config.moe_config.top_k_experts,
                    capacity_factor=self.config.moe_config.capacity_factor,
                    eval_capacity_factor=self.config.moe_config.eval_capacity_factor,
                    min_capacity=self.config.moe_config.min_capacity,
                    use_residual=self.config.moe_config.use_residual,
                )
                self._validate_experts_weights(pretrained_state_dict, layer_num)

            elif self.config.moe_config.moe_class == "deepseek":
                self.language_model.model.layers[layer_num].mlp = DeepseekV2MoE(
                    num_experts_per_tok=self.config.moe_config.top_k_experts,
                    ep_size=self.config.moe_config.ep_size,
                    n_routed_experts=num_experts,
                    shared_experts=self.config.moe_config.use_residual,
                    expert=expert,
                    routed_scaling_factor=1,
                    scoring_func="softmax",
                    aux_loss_alpha=1,
                    seq_aux=False,
                    topk_method="gready",
                    n_group=None,
                    topk_group=None,
                    norm_topk_prob=self.config.norm_topk_prob,
                    hidden_size=self.config.text_config.hidden_size,
                    
                )
                self._validate_experts_weights(pretrained_state_dict, layer_num)

    def _validate_experts_weights(self, pretrained_state_dict, layer_num):
        """
        Ensure that the weights of the newly created experts match the pretrained ones.
        """
        if isinstance(self.language_model.model.layers[layer_num].mlp, deepspeed.moe.layer.MoE):
            experts = self.language_model.model.layers[layer_num].mlp.deepspeed_moe.experts.deepspeed_experts
        else:
            experts = self.language_model.model.layers[layer_num].mlp.experts
            
        for e in experts:
            loaded_state_dict = e.state_dict()
            assert all([torch.allclose(pretrained_state_dict[k], v) for k, v in loaded_state_dict.items()])
            assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict.items()])

    def _update_forward_functions(self):
        """
        Replace the forward methods of the model layers and the model itself.
        """
        MoEDecoderForward, MoEModelForward = FORWARD_MAP[self.config.language_backbone_name]
        
        # Update decoder layer forward methods
        for m in self.language_model.model.layers:
            m.forward = MoEDecoderForward(m)
        logger.info(f'Replaced DecoderLayer.forward with MoEDecoderLayer.forward', on_rank0=True)
        
        # Update model forward method
        self.language_model.model.forward = MoEModelForward(self.language_model.model)
        logger.info(f'Replaced Model.forward with MoEModel.forward', on_rank0=True)

    def _log_moe_configuration(self, num_layers, moe_layers_idx):
        """
        Log the MoE layer configuration.
        """
        moe_config_str = "\n".join([
            f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where",
            *[f'layer-{layer_num} has {num_experts} experts' for num_experts, layer_num in zip(self.config.moe_config.num_experts, moe_layers_idx)]
        ])
        logger.info(moe_config_str, on_rank0=True)


    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def encode_mol_v2(self, mol, device) -> torch.Tensor:
        _, h_node = self.graph_tower.float().encode_mol(mol, proj=False, return_node_feats=True)
        dtype = h_node.dtype
            
        if dtype == torch.bfloat16:
            h_node = h_node.float()
                
        graph_features = self.mm_projector.float()(h_node)

        return graph_features.to(self.language_model.dtype)
    
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.Tensor, 
        past_key_values: list[torch.FloatTensor], 
        labels: torch.LongTensor, 
        graphs: torch.FloatTensor,
    ):
        if graphs is None or input_ids.shape[1] == 1:
            # In case input_ids.shape[1] == 1 & graphs==None & past_key_values != None, we are in the case of
            # generation with cache
            if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                # Structure of kv cache:
                # tuple[tuple[torch.Tensor]]
                # first tuple: layers
                # second tuple: K, V
                # torch.Tensor: B, num_head, cache_length, head_dim
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                
            return None, labels, input_ids, attention_mask, past_key_values, None
        
        
        # 1. embed graph into graph tokens
        # check batch size misalignment
        assert len(graphs) == input_ids.shape[0], f"batch size misaligned! graphs: {len(graphs)}, texts: {input_ids.shape[0]}"
        # embed all graphs
        if not isinstance(graphs, list):
            graphs = graphs.to_data_list()
        graph_features = [self.encode_mol_v2(graph, input_ids.device) if graph is not None else torch.Tensor([]) for graph in graphs]

        # 2. Pre-allocate new data according to the graph token length
        # graph_featires: list[N x D]
        len_graph_tokens = [feat.shape[0] for feat in graph_features]
        max_graph_len = max(len_graph_tokens)
        if max_graph_len == 0:
            max_graph_len = 1  # in case the batch is pure text, then we will mis calculate because there is no <image> token
        # The seq len of fused embedding should be the max_len + seq len of input_ids
        # -1 because <image> token took one place, we will remove it, so naturally we need graph_token_len-1 new positions.
        
        dtype = next(self.language_model.model.embed_tokens.parameters()).dtype
        device = next(self.language_model.model.embed_tokens.parameters()).device
        fused_embeds = torch.zeros(
            size=[input_ids.shape[0], input_ids.shape[1]+max_graph_len-1, self.config.text_config.hidden_size],
            dtype=dtype,
            device=device
        )
        fused_attention_mask = torch.zeros(
            size=[input_ids.shape[0], input_ids.shape[1]+max_graph_len-1],
            dtype=attention_mask.dtype,
            device=attention_mask.device
        ).fill_(False)
        
        if labels is not None:
            fused_labels = torch.zeros(
                size=[input_ids.shape[0], input_ids.shape[1]+max_graph_len-1],
                dtype=labels.dtype,
                device=labels.device
            ).fill_(IGNORE_INDEX)
        else:
            fused_labels = None

        # 3. Insert
        """
        Bubbles in the input
        [
            [1, 2, 3, 4, p, p],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, p, p, p]
        ]
        and the longest graph feature is [g, g, g, g], len=4, for the last sample
        then the allocated version
        [
            [1, 2, 3, g, 4, p, p, p, p, p]
            [1, 2, 3, g, 4, 5, 6, p, p, p]
            [1, 2, g, g, g, g, 3, p, p, p]
        ]
        We actually only need
        [
            [1, 2, 3, g, 4, p, p],
            [1, 2, 3, g, 4, 5, 6]
            [1, 2, g, g, g, g, 3]
        ]
        This is because the longest sequence changed
        """
        # TODO: What about multiple <image> tokens?
        # TODO: Is it possible to use pure tensor operations? So we can fully parallel...
        for batch_idx in range(input_ids.shape[0]):
            # get position of <image>
            mm_pos = torch.where(input_ids[batch_idx] == self.config.image_token_index)[0]
            # no <image> token in this batch, we do not need to insert
            if len(mm_pos) == 0:
                assert graph_features[batch_idx].shape[0] == 0, f"You sure you don't add graph features?"
                fused_embeds[batch_idx, :input_ids[batch_idx].shape[0], :] = self.language_model.model.embed_tokens(input_ids[batch_idx])
                if labels is not None:
                    fused_labels[batch_idx, :input_ids[batch_idx].shape[0]] = labels[batch_idx]
                fused_attention_mask[batch_idx, :input_ids[batch_idx].shape[0]] = attention_mask[batch_idx]
                continue
                
            # we got mutli-modal input
            # a. get text embeddings from left side and right side of <image>
            left_side_ids, right_side_ids = input_ids[batch_idx, :mm_pos], input_ids[batch_idx, mm_pos+1:]
            # b. embed them
            left_side_text_embedding = self.language_model.model.embed_tokens(left_side_ids)
            right_side_text_embedding = self.language_model.model.embed_tokens(right_side_ids)
            # c. fill into embeddings
            fused_embeds[batch_idx, :mm_pos, :] = left_side_text_embedding
            fused_embeds[batch_idx, mm_pos:mm_pos+len_graph_tokens[batch_idx], :] = graph_features[batch_idx]
            fused_embeds[batch_idx, mm_pos+len_graph_tokens[batch_idx]:mm_pos+len_graph_tokens[batch_idx]+len(right_side_ids), :] = right_side_text_embedding
            # d. fill into labels
            if labels is not None:
                left_side_labels, right_side_labels = labels[batch_idx, :mm_pos], labels[batch_idx, mm_pos+1:]
                fused_labels[batch_idx, :mm_pos] = left_side_labels
                fused_labels[batch_idx, mm_pos:mm_pos+len_graph_tokens[batch_idx]] = torch.full((len_graph_tokens[batch_idx],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
                fused_labels[batch_idx, mm_pos+len_graph_tokens[batch_idx]:mm_pos+len_graph_tokens[batch_idx]+len(right_side_labels)] = right_side_labels
            # e. fill into attention masks
            left_side_mask, right_side_mask = attention_mask[batch_idx, :mm_pos], attention_mask[batch_idx, mm_pos+1:]
            fused_attention_mask[batch_idx, :mm_pos] = left_side_mask
            fused_attention_mask[batch_idx, mm_pos:mm_pos+len_graph_tokens[batch_idx]] = torch.full((len_graph_tokens[batch_idx],), True, device=attention_mask.device, dtype=attention_mask.dtype)
            fused_attention_mask[batch_idx, mm_pos+len_graph_tokens[batch_idx]:mm_pos+len_graph_tokens[batch_idx]+len(right_side_mask)] = right_side_mask
            
        # logger.info(f"before: {fused_labels}, len={fused_labels.shape[1]}")
        # Remove bubbles in the input
        # 1. find max_len, hope, we don't have False before pad tokens...
        max_original_len = fused_attention_mask.sum(dim=1).max()
        # 2. truncate
        fused_embeds = fused_embeds[:, :max_original_len, :]
        if labels is not None:
            fused_labels = fused_labels[:, :max_original_len]
        fused_attention_mask = fused_attention_mask[:, :max_original_len]
        
        position_ids = (fused_attention_mask.cumsum(-1) - 1).masked_fill_((fused_attention_mask == 0), 1)
        if fused_embeds.shape[1] > self.config.text_config.max_position_embeddings:
            logger.warning(f"Input embedding after multimodal insertion is too long. {fused_embeds.shape[1]} > {self.config.text_config.max_position_embeddings}")
    
        return fused_embeds, fused_labels, None, fused_attention_mask, None, position_ids
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graphs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds, labels, input_ids, attention_mask, past_key_values, position_ids = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, graphs)

        outputs = self.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        
        logits = self.language_model.lm_head(hidden_states)
        
        loss = None
        model_loss = None
        moe_loss = None
        moe_loss_list = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            model_loss = loss.detach().clone()
        
        
            if len(outputs[-1]) > 0 and isinstance(outputs[-1], list):
                moe_loss_list = outputs[-1]
                moe_losses = [moe_loss for moe_loss in moe_loss_list if moe_loss is not None]
                moe_loss = self.config.moe_config.router_aux_loss_coef * sum(moe_losses)
                loss += moe_loss
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output
        
        return_class = MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=sum(moe_losses) if moe_loss is not None else None,
            model_loss=model_loss,
            logits=logits,
            router_aux_coeff=getattr(self.config.moe_config, "router_aux_loss_coef", None),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_loss_list=moe_loss_list,
            attention_mask=attention_mask
        )
        
        return return_class
        
    """
    NOTE
    Borrowed code from transformers
    """
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
            
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graphs": kwargs.get("graphs", None)
            }
        )
        
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
