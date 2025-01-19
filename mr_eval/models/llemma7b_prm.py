import hashlib
import json
import os
from typing import List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate import Accelerator


from .abstract_model import prm
from ..utils.utils import *

class Llemma7bPRM(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/llemma-7b-prm-prm800k-level-1to3-hf",
            tokenizer_pretrained = "EleutherAI/llemma_7b",
            step_tag = "\n\n",
            validity_threshold = 0.5,
        ) -> None:
        
        super().__init__(validity_threshold=validity_threshold)
        
        self.step_tag = step_tag
        
        self.model = AutoModelForCausalLM.from_pretrained(
                                                pretrained, 
                                            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()
        
        


    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        res = []
        for idx,step in enumerate(steps):
            if "\n\n" in step:
                step = step.replace("\n\n", "")
            res.append(f"{step.strip()}{self.step_tag}")
        steps_str = "".join(res)
        original_input_for_prm = f"# Question\n\n{question}\n\n# Solution\n\n{steps_str}"
        begin_solution_tokens = self.tokenizer.encode("\n\n# Solution", add_special_tokens=False)[1:]
        scoring_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)[1:]
        eos_token = self.tokenizer.eos_token_id

        input_ids = self.tokenizer.encode(original_input_for_prm)

        begin_solution_flag = False
        candidate_positions = []
        for start_idx in range(len(input_ids)):
            if tuple(input_ids[start_idx:start_idx+len(begin_solution_tokens)]) == tuple(begin_solution_tokens):
                begin_solution_flag = True

            if begin_solution_flag and tuple(input_ids[start_idx:start_idx+len(scoring_tokens)]) == tuple(scoring_tokens):
                candidate_positions.append(start_idx)

            if input_ids[start_idx] == eos_token:
                candidate_positions.append(start_idx)
                break

        # maybe delete the first and the second to last candidate_positions
        # because they are "\n\n" after "# Solution" and after "# Answer"
        del candidate_positions[0]
        
        input_ids = torch.tensor(input_ids)
        candidate_positions = [i for i in candidate_positions if i < self.generation_config.max_length]
        candidate_positions = torch.tensor(candidate_positions)
        res = dict(
            idx = data_idx,
            input_ids = input_ids,
            candidate_positions = candidate_positions,
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:
        
        self.model, dataloader = self.accelerator.prepare(self.model, dataloader)
        self.accelerator.wait_for_everyone()
        self.model.eval()
        gen_kwargs = dataloader.dataset.gen_kwargs
        progress_bar = tqdm_rank0(len(dataloader), desc="Model Responding")
        if len(dataloader) == 0:
            self.accelerator.wait_for_everyone()
            return
        with torch.no_grad():
            for batch in dataloader:
                idx = batch['idx']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                candidate_positions = batch['candidate_positions']
                
                original_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                scores = original_logits.mean(dim=-1)
                
                for i in range(len(idx)):
                    current_candidate_positions = candidate_positions[i]
                    current_score = scores[i][current_candidate_positions]
                    
                    original_step_scores = torch.sigmoid(current_score).tolist()
                    step_level_validity_labels = [item > self.validity_threshold for item in original_step_scores]
                    idx_item = idx[i]
                    score_dict = dict(
                        step_level_validity_labels = step_level_validity_labels,
                        step_level_validity_scores = original_step_scores,
                    )
                    res = dict(scores=score_dict, idx=idx_item)
                    dataloader.dataset.store_results(res)
                if progress_bar is not None:
                    progress_bar.update(1)
        
        self.accelerator.wait_for_everyone()