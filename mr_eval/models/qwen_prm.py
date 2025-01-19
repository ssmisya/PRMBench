import abc
import hashlib
import json
import os
import logging
from typing import List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoTokenizer,MistralModel, MistralPreTrainedModel, LlamaModel, LlamaPreTrainedModel, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from accelerate import Accelerator

from .abstract_model import prm
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ..utils.model_utils import remove_step_prefix
from accelerate.logging import get_logger as get_accelerator_logger

# accelerate_logger = logging.getLogger("debug")
# accelerate_logger.setLevel(logging.DEBUG)
logger = get_logger(__name__)
class QwenPRM(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/Qwen2.5-Math-PRM-7B",
            redundancy_threshold = 0.15,
            validity_threshold = 0.5,
        ) -> None:
        super().__init__(validity_threshold=validity_threshold, redundancy_threshold=redundancy_threshold)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            pretrained, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
                
        self.accelerator = Accelerator()
        
        
        self.step_separator = "<extra_0>"
        self.step_separator_token_id = self.tokenizer.encode(self.step_separator)[0]

    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        
        
        ## build model-specialized input
        system_prompt = "Please reason step by step, and put your final answer within \boxed{}."
        combined_steps = ""
        for step in steps:
            cleaned_step = remove_step_prefix(step)
            combined_steps += cleaned_step + self.step_separator
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": combined_steps},
        ]
        
        conversation_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        input_ids = self.tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
        )
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
            for batch_idx, batch in enumerate(dataloader):
                
                idx = batch['idx']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                # print(f"data device: {input_ids.device}, current device: {self.accelerator.device}")
                scores = self.model(input_ids,
                                    attention_mask,).logits
                token_mask = input_ids == self.step_separator_token_id
                step_reward = make_step_rewards(scores, token_mask)
                # import debugpy
                # debugpy.listen(address = ('0.0.0.0', 7119))
                # debugpy.wait_for_client() 
                # breakpoint() #在下一句代码处暂停  
                
                
                for i in range(len(idx)):
                    idx_item = idx[i]
                    try:
                        step_level_validity_scores = step_reward[i]
                        score_dict = dict(
                            step_level_validity_scores=step_level_validity_scores,
                            step_level_validity_labels=[item > self.validity_threshold for item in step_level_validity_scores],
                        )
                        res = dict(scores=score_dict, idx=idx_item)
                    except:
                        logger.error(f"Error in processing idx: {idx[i]}")
                        res = dict(scores=dict(), idx=idx_item,validity=False)
                        
                    dataloader.dataset.store_results(res)
                if progress_bar is not None:
                    progress_bar.update(1)
        
        self.accelerator.wait_for_everyone()
        
        
def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res