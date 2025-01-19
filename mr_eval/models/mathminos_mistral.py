import hashlib
import json
import os
from typing import List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoTokenizer,MistralForTokenClassification
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate import Accelerator
import torch.nn.functional as F

from .abstract_model import prm
from ..utils.utils import *
from ..utils.model_utils import remove_step_prefix

class MathMinos_Mistral(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/ref/MATH-Minos/RM/ckpts/minos_mistral",
            step_tag = "и",
            validity_threshold = 0,
        ) -> None:
        
        super().__init__(validity_threshold=validity_threshold)
        self.step_tag = step_tag
        
        self.model = MistralForTokenClassification.from_pretrained(
                                                pretrained, 
                                            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1]
        if self.tokenizer.pad_token_id is None:
            print_rank0("Setting pad_token_id to eos_token_id")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()
        
        


    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        res = []
        for idx,step in enumerate(steps):
            clean_step = remove_step_prefix(step)
            res.append(f"Step {idx+1}: {clean_step} {self.step_tag}\n")
            
        steps_str = "".join(res)
        original_input_for_prm = f"Human: {question}\n\nAssistant:{steps_str}"

        input_ids = self.tokenizer.encode(original_input_for_prm, return_tensors='pt', max_length = self.generation_config.max_length, truncation=True)
        while input_ids.ndim > 1:
            input_ids = input_ids[0]
            
        res = dict(
            idx = data_idx,
            input_ids = input_ids,
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
                
                original_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits.squeeze(-1)
                
                for i in range(len(idx)):
                    current_input_ids = input_ids[i]
                    current_logits = original_logits[i][current_input_ids == self.step_tag_id]
                    original_labels = current_logits > self.validity_threshold
                    if torch.is_tensor(original_labels):
                        original_labels = original_labels.tolist()
                    # original_scores = F.normalize(original_scores, p=2, dim=-1)
                    original_scores = current_logits
                    idx_item = idx[i]
                    score_dict = dict(
                        step_level_validity_labels = original_labels,
                        step_level_validity_scores = original_scores.tolist(),
                    )
                    res = dict(scores=score_dict, idx=idx_item)
                    dataloader.dataset.store_results(res)

                if progress_bar is not None:
                    progress_bar.update(1)
        
        self.accelerator.wait_for_everyone()