import abc
import hashlib
import json
import os
from typing import List, Optional, Tuple
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoTokenizer,MistralModel, MistralPreTrainedModel, LlamaModel, LlamaPreTrainedModel, AutoTokenizer
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate import Accelerator

from .abstract_model import prm
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ..utils.model_utils import remove_step_prefix


logger = get_logger(__name__)
class ReasonEval(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/ReasonEval-7B",
            model_size = "7B",
            redundancy_threshold = 0.15,
            validity_threshold = 0.5,
        ) -> None:
        super().__init__(validity_threshold=validity_threshold, redundancy_threshold=redundancy_threshold)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        if model_size == '7B':
            model = ReasonEval_7B.from_pretrained(pretrained,torch_dtype=torch.bfloat16,)
        elif model_size == '34B':
            model = ReasonEval_34B.from_pretrained(pretrained,torch_dtype=torch.bfloat16,)
        else:
            raise ValueError(f"Invalid model size: {model_size}")
        self.tokenizer = tokenizer
        self.model = model
        self.origin_model = model
        self.accelerator = Accelerator()
        


    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        ## build model-specialized input
        PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        step_separator = f"{self.tokenizer.pad_token}"
        combined_steps = ""
        for step in steps:
            cleaned_step = remove_step_prefix(step)
            combined_steps += cleaned_step + step_separator
        prompt = PROMPT_FORMAT.format(input = question)
        tokenized_result = self.tokenizer(prompt + step_separator + combined_steps)['input_ids']
        
        ## Separating labels and adjusting token IDs
        separator_token_id = self.tokenizer(step_separator)['input_ids'][-1]
        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)
        if isinstance(self.origin_model,ReasonEval_7B):
            adjusted_token_ids = [1] + adjusted_token_ids # Adjusting to recover the first token_ids of the sentences
            adjusted_token_ids=torch.tensor(adjusted_token_ids)
            labeled_token_indices = labeled_token_indices[2:]  # Adjusting to skip the first two separator (begining and endding of the problems)
        elif isinstance(self.origin_model,ReasonEval_34B):
            adjusted_token_ids=torch.tensor(adjusted_token_ids)
            labeled_token_indices = labeled_token_indices[1:]  # Adjusting to skip the first separator (endding of the problems)
        else:
            raise ValueError(f"Invalid model size!")
        
        assert len(labeled_token_indices) == len(steps), f"len(labeled_token_indices): {len(labeled_token_indices)}, len(steps): {len(steps)}"
        labeled_token_indices = [i for i in labeled_token_indices if i < self.generation_config.max_length]
        assert adjusted_token_ids.ndim == 1
        res = dict(
            idx = data_idx,
            input_ids = adjusted_token_ids,
            labeled_token_indices = labeled_token_indices,
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:
        self.model, dataloader = self.accelerator.prepare(self.model, dataloader)
        self.origin_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.wait_for_everyone()
        self.model.eval()
        gen_kwargs = dataloader.dataset.gen_kwargs
        progress_bar = tqdm_rank0(len(dataloader), desc="Model Responding")
        if len(dataloader) == 0:
            self.accelerator.wait_for_everyone()
            return
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                idx = batch['idx']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labeled_token_indices = batch['labeled_token_indices']

                scores = self.model(input_ids,
                                    attention_mask,)     
                for i in range(len(idx)):
                    idx_item = idx[i]
                    try:
                        score = scores[i, labeled_token_indices[i], :]
                        score = torch.softmax(score, dim=-1).tolist()
                        step_level_validity_scores =  [(score_item[1] + score_item[2]) for score_item in score]
                        step_level_redundancy_scores = [score_item[1] for score_item in score]
                        score_dict = dict(step_level_redundancy_scores=step_level_redundancy_scores, 
                            step_level_validity_scores=step_level_validity_scores,
                            step_level_redundancy_labels=[item > self.redundancy_threshold for item in step_level_redundancy_scores], 
                            step_level_validity_labels=[item > self.validity_threshold for item in step_level_validity_scores],
                            solution_level_redundancy_scores= max(step_level_redundancy_scores), 
                            solution_level_validity_scores=min(step_level_validity_scores),
                            solution_level_redundancy_labels=max(step_level_redundancy_scores)>self.redundancy_threshold,
                            solution_level_validity_labels=min(step_level_validity_scores)>self.validity_threshold)
                        res = dict(scores=score_dict, idx=idx_item)
                    except:
                        logger.error(f"Error in processing idx: {idx[i]}")
                        res = dict(scores=dict(), idx=idx_item,validity=False)
                        
                    dataloader.dataset.store_results(res)
                if progress_bar is not None:
                    progress_bar.update(1)
        
        self.accelerator.wait_for_everyone()

    
        
    
    

def reasoneval_inference(question,reasoning_steps,model,tokenizer):
    # Preprocessing input
    PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
    step_separator = f"{tokenizer.pad_token}"
    combined_steps = ""
    for steps in reasoning_steps:
        combined_steps += steps + step_separator
    prompt = PROMPT_FORMAT.format(input=question)
    tokenized_result = tokenizer(prompt + step_separator + combined_steps)['input_ids']

    ## Separating labels and adjusting token IDs
    separator_token_id = tokenizer(step_separator)['input_ids'][-1]
    labeled_token_indices = []
    adjusted_token_ids = []
    separator_count = 0
    for idx, token_id in enumerate(tokenized_result):
        if token_id == separator_token_id:
            labeled_token_indices.append(idx - 1 - separator_count)
            separator_count += 1
        else:
            adjusted_token_ids.append(token_id)
    if isinstance(model,ReasonEval_7B):
        adjusted_token_ids = [1] + adjusted_token_ids # Adjusting to recover the first token_ids of the sentences
        adjusted_token_ids=torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[2:]  # Adjusting to skip the first two separator (begining and endding of the problems)
    elif isinstance(model,ReasonEval_34B):
        adjusted_token_ids=torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[1:]  # Adjusting to skip the first separator (endding of the problems)
    else:
        raise ValueError(f"Invalid model size!")
    assert len(labeled_token_indices) == len(reasoning_steps)

    attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
    # Evaluating reasoning steps using ReasonEval
    with torch.no_grad():
        adjusted_token_ids = adjusted_token_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        reasoning_scores = model(adjusted_token_ids, attention_mask)[0,labeled_token_indices , :]
        scores = torch.softmax(reasoning_scores, dim=-1).tolist()

    # Calculating the validity and redundancy scores
    ## score: [p_{negative}, p_{neutral}, p_{positive}]

    ## S_{validity} = p_{neutral} + p_{positive}
    step_level_validity_scores =  [(score[1] + score[2]) for score in scores]
    # print(f"step_level_validity_scores: {step_level_validity_scores}")
    # ReasonEval-7B for the example: [0.9492, 0.7863 ,0.2520, 0.7860, 0.9125, 0.5916, 0.4494, 0.8189, 0.8240, 0.2671]
    # ReaspnEval-34B for the example: [0.9360, 0.6813 ,0.0720, 0.2811, 0.4531, 0.1122, 0.1328, 0.2026, 0.2265 0.1163]

    ## S_{redundancy} = p_{neutral}
    step_level_redundancy_scores = [score[1] for score in scores]
    # print(f"step_level_redundancy_scores: {step_level_redundancy_scores}")
    # ReasonEval-7B for the example: [0.4433, 0.1287, 0.0397, 0.0789, 0.0789, 0.0509, 0.0487, 0.0702, 0.0955, 0.0120]
    # ReasonEval-34B for the example: [0.6060, 0.1682, 0.0258, 0.1044, 0.1604, 0.0404, 0.0447, 0.0507, 0.0492, 0.0236]
    
    solution_level_validity_scores = min(step_level_validity_scores)
    # print(f"solution_level_validity_scores: {solution_level_validity_scores}")
    solution_level_redundancy_scores = max(step_level_redundancy_scores)
    # print(f"solution_level_validity_scores: {solution_level_redundancy_scores}")
    return step_level_validity_scores,step_level_redundancy_scores,solution_level_validity_scores,solution_level_redundancy_scores

class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()  # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores
    
class ReasonEval_34B(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
      
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores