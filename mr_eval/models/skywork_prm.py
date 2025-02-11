import hashlib
import json
import os
import sys

from typing import List, Optional, Tuple, Type, TypeVar, Union

from tqdm import tqdm
from transformers import AutoTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate import Accelerator


from .abstract_model import prm
from ..utils.utils import *
from ..utils.log_utils import *
from ..utils.model_utils import remove_step_prefix

logger = get_logger(__name__)

try:
    # Some skywork specific functions.
    # please modify your code if you are evaluating skyworkPRM
    sys.path.append('/mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/ref/skywork-o1-prm-inference')
    from model_utils.prm_model import PRM_MODEL
    from model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards
except:
    logger.error("Failed to import Skywork PRM model utils, please specify path to Skywork PRM.")
    

    
class SkyworkPRM(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
            step_tag = 'ки',
            validity_threshold = -0.05,
        ) -> None:
        
        super().__init__(validity_threshold=validity_threshold)
        self.step_tag = step_tag

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = PRM_MODEL.from_pretrained(pretrained, device_map="cpu").eval()
        self.accelerator = Accelerator()
        assert not self.accelerator.state.distributed_type == 'DEEPSPEED', "DeepSpeed is not supported for Skywork PRM."

    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        res = []
        for idx,step in enumerate(steps):
            clean_step = remove_step_prefix(step)
            res.append(f"Step {idx+1}: {clean_step.strip()} {self.step_tag}\n")

        steps_str = "".join(res)
        prm_input_data = dict(
            problem = question,
            response = steps_str,
        )
        input_ids, attention_mask, reward_flags = prepare_input(**prm_input_data,tokenizer=self.tokenizer,step_token=self.step_tag)
        input_ids, reward_flags = torch.LongTensor(input_ids), torch.LongTensor(reward_flags)
        while input_ids.ndim > 1:
            input_ids = input_ids[0]
        if input_ids.shape[0] > self.generation_config.max_length:
            input_ids = input_ids[:self.generation_config.max_length]
        if reward_flags.shape[0] > self.generation_config.max_length:
            reward_flags = reward_flags[:self.generation_config.max_length]
        res = dict(
            idx = data_idx,
            input_ids = input_ids,
            reward_flags = reward_flags,
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:
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
                attention_mask = batch['attention_mask']
                reward_flags = batch['reward_flags']
                
                _, _, rewards = self.model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
                step_rewards = derive_step_rewards(rewards, reward_flags)
                
                
                for i in range(len(idx)):
                    step_level_validity_scores = step_rewards[i]
                    judge_label_scores = [step_level_validity_scores[0]] + [step_level_validity_scores[i] - step_level_validity_scores[i-1] for i in range(1, len(step_level_validity_scores))]
                    step_level_validity_labels = [item > self.validity_threshold for item in judge_label_scores]

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