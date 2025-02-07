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
from copy import deepcopy



from .abstract_model import prm
from ..utils.prompts import PROMPT_DICT
from ..utils.utils import *
from ..utils.log_utils import *
from ..utils.model_utils import remove_step_prefix, process_policy_lm_evaluation_response

logger = get_logger(__name__)


    
class QwenQwQ(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/QwQ-32B-Preview",
            validity_threshold = 0,
            redundancy_threshold = 0,
            log_save_dir = "mr_eval/scripts/logs/generated/qwq.jsonl",
        ) -> None:
        super().__init__(validity_threshold=validity_threshold, redundancy_threshold=redundancy_threshold)
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(pretrained,).eval()
        self.log_save_dir = log_save_dir
        
        self.accelerator = Accelerator()
        self.model = self.model.to(self.accelerator.device)
        self.prompt = PROMPT_DICT["policy_model_as_an_evaluator"]
        self.messages = [
            {"role": "system", "content": self.prompt["system_prompt"]},
            {"role": "user", "content": self.prompt["fewshots"][0][0]},
            {"role": "assistant", "content": self.prompt["fewshots"][0][1]},
            {"role": "user", "content": self.prompt["fewshots"][1][0]},
            {"role": "assistant", "content": self.prompt["fewshots"][1][1]},
        ]


    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        res = []
        for idx,step in enumerate(steps):
            clean_step = remove_step_prefix(step)
            res.append(f"Step {idx+1}: {clean_step} \n\n")
            
        steps_str = "".join(res)
        original_input_for_prm = f"Question: {question}\n\n Solutions: {steps_str}"
        messages = deepcopy(self.messages)
        messages.append({"role": "user", "content": original_input_for_prm})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        while input_ids.ndim > 1:
            input_ids = input_ids[0]
            
        res = dict(
            idx = data_idx,
            input_ids = input_ids,
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:
        dataloader = self.accelerator.prepare(dataloader)
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
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1024,
                )
                generated_ids = [
                    output_id[len(input_id):] for input_id, output_id in zip(input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
   
                for i in range(len(idx)):
                    try:
                        current_idx = idx[i]
                        current_response = response[i]
                        scores = process_policy_lm_evaluation_response(current_response)
                        if scores:
                            score_dict = dict(
                                step_level_validity_scores = scores["validity"],
                                step_level_redundancy_scores = scores["redundancy"],
                                step_level_validity_labels = [item > 0 for item in scores["validity"]],
                                step_level_redundancy_labels = [item > 0 for item in  scores["redundancy"]],
                            )
                            res = dict(scores=score_dict, idx=current_idx, validity=True)
                        else:
                            res = dict(validity=False, idx=current_idx)
                        dataloader.dataset.store_results(res)
                        log = dict(idx = current_idx, response = current_response, scores = scores, result = res)
                        dataloader.dataset.save_result_item_into_log(log,self.log_save_dir)
                    except:
                        current_idx = idx[i]
                        current_response = response[i]
                        logger.error(f"Error in responding to idx {current_idx}")
                        res = dict(validity=False, idx=current_idx)
                        dataloader.dataset.store_results(res)
                        log = dict(idx = current_idx, response = current_response, scores = None, result = res)
                        dataloader.dataset.save_result_item_into_log(log,self.log_save_dir)
                if progress_bar is not None:
                    progress_bar.update(1)
        
        self.accelerator.wait_for_everyone()