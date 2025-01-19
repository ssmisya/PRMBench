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
class MathShepherd(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/math-shepherd-mistral-7b-prm",
            good_token = '+',
            bad_token = '-',
            step_tag = 'ки',
            validity_threshold = 0.5,
        ) -> None:
        
        super().__init__(validity_threshold=validity_threshold)
        self.good_token = good_token
        self.bad_token = bad_token
        self.step_tag = step_tag

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902
        self.model = AutoModelForCausalLM.from_pretrained(pretrained,).eval()

        self.accelerator = Accelerator()
        
        


    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        res = []
        for idx,step in enumerate(steps):
            if step.strip().startswith("Step") or step.strip().startswith("step"):
                res.append(f"{step.strip()} {self.step_tag}\n")
            else:
                res.append(f"Step {idx+1}: {step.strip()} {self.step_tag}\n")
        steps_str = "".join(res)
        original_input_for_prm = f"{question} {steps_str}"
        input_ids=self.tokenizer.encode(original_input_for_prm, return_tensors='pt')
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
                ).logits
                
                for i in range(len(idx)):
                    current_logits = original_logits[i][:,self.candidate_tokens]
                    original_scores = current_logits.softmax(dim=-1)[:,0] 
                    original_step_scores = original_scores[input_ids[i] == self.step_tag_id].tolist()
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