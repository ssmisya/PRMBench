import hashlib
import json
import os
import sys

from typing import List, Optional, Tuple, Type, TypeVar, Union

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate import Accelerator


from .abstract_model import prm
from ..utils.utils import *
from ..utils.log_utils import *
from ..utils.model_utils import remove_step_prefix, find_subsequence

logger = get_logger(__name__)


    
class LLaMA318BPRM(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/Llama3.1-8B-PRM-Mistral-Data",
            positive_token = 10,
            pattern = torch.tensor([128006,  78191, 128007,    271,     10, 128009]),
            validity_threshold = 0.5,
        ) -> None:
        super(LLaMA318BPRM, self).__init__(validity_threshold=validity_threshold)
        
        # pattern
        if isinstance(pattern, torch.Tensor):
            self.pattern = pattern
        elif isinstance(pattern, list):
            self.pattern = torch.tensor(pattern)
        else:
            raise ValueError("pattern should be a list or a torch.Tensor")
        self.positive_token = positive_token
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="cpu").eval()
        
        self.accelerator = Accelerator()
        
        


    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        conversation = []
        for idx,step in enumerate(steps):
            clean_step = remove_step_prefix(step)
            if idx == 0:
                text = question + " " + clean_step
            else:
                text = clean_step
            conversation.append({"content":text,"role":"user"})
            conversation.append({"content":"+","role":"assistant"})
        input_ids = self.tokenizer.apply_chat_template(conversation,return_tensors="pt")

        while input_ids.ndim > 1:
            input_ids = input_ids[0]
        res = dict(
            idx = data_idx,
            input_ids = input_ids,
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:
        # self.model, dataloader = self.accelerator.prepare(self.model, dataloader)
        dataloader = self.accelerator.prepare(dataloader)
        self.model = self.model.to(self.accelerator.device)
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
                logits = self.model(input_ids).logits
                
                for i in range(len(idx)):
                    current_input_id = input_ids[i].cpu()
                    current_logits = logits[i].cpu()
                    score_locations = find_subsequence(current_input_id, self.pattern)
                    score_locations = [i+3 for i in score_locations]
                    reward_logits = current_logits[score_locations]
                    step_level_validity_scores = reward_logits.softmax(dim=-1)[:,self.positive_token].tolist()
                    step_level_validity_labels = [item > self.validity_threshold for item in step_level_validity_scores]

                    idx_item = idx[i]
                    score_dict = dict(
                        step_level_validity_labels = step_level_validity_labels,
                        step_level_validity_scores = step_level_validity_scores,
                    )
                    res = dict(scores=score_dict, idx=idx_item)
                    dataloader.dataset.store_results(res)
                if progress_bar is not None:
                    progress_bar.update(1)
        
        self.accelerator.wait_for_everyone()