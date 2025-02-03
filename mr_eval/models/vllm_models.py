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
from vllm import LLM, SamplingParams


from .abstract_model import prm
from ..utils.prompts import PROMPT_DICT
from ..utils.utils import *
from ..utils.log_utils import *
from ..utils.model_utils import remove_step_prefix, process_policy_lm_evaluation_response
import torch.multiprocessing as mp

# 设置启动方法为 spawn
mp.set_start_method('spawn', force=True)
logger = get_logger(__name__)


    
class VllmModels(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/QwQ-32B-Preview",
            tensor_parallel: str = "1",
            validity_threshold = 0,
            redundancy_threshold = 0,
            first_round_role = "user",
            save_to_ckpt_interval = 1000,
        ) -> None:
        super().__init__(validity_threshold=validity_threshold, redundancy_threshold=redundancy_threshold)
        
        self.tensor_parallel = int(tensor_parallel)
        self.first_round_role = first_round_role
        self.save_to_ckpt_interval = save_to_ckpt_interval
        pid = os.getpid()
        # print(f"当前进程 ID: {pid}")
        
        self.model = LLM(model=pretrained, tensor_parallel_size = self.tensor_parallel, trust_remote_code=True)
        self.prompt = PROMPT_DICT["policy_model_as_an_evaluator"]
        self.messages = [
            {"role": self.first_round_role, "content": self.prompt["system_prompt"]},
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

            
        res = dict(
            idx = data_idx,
            inputs = messages,
            model_type = "vllm",
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:

        # gen_kwargs = dataloader.dataset.gen_kwargs
        sampling_params = SamplingParams(
            temperature = self.generation_config.get("temperature", 0.0),
            top_k = self.generation_config.get("top_k", -1),
            top_p = self.generation_config.get("top_p", 1.0),
            max_tokens = self.generation_config.get("max_length", 2048),
        )
        
        
        progress_bar = tqdm_rank0(len(dataloader), desc="Model Responding")
        
        dataloader_iter = iter(dataloader)
        # import debugpy
        # debugpy.listen(address = ('0.0.0.0', 7119))
        # debugpy.wait_for_client() 
        # breakpoint() # 在下一句代码处暂停
        # dist.barrier()
        with torch.no_grad():
            stop_flag = False
            while not stop_flag:
                data_batch = []
                messages = []
                idxs = []
                for iter_num in range(self.save_to_ckpt_interval):
                    try:
                        current_batch = next(dataloader_iter)
                        message = current_batch['inputs']
                        idx = current_batch['idx']
                        messages.append(message)
                        idxs.append(idx)
                    except StopIteration:
                        stop_flag = True
                        break
                    
                outputs = self.model.chat(messages, sampling_params = sampling_params)
                
                for idx, output in zip(idxs, outputs):
                    response = output.outputs[0].text
                    try:
                        scores = process_policy_lm_evaluation_response(response)
                        if scores:
                            score_dict = dict(
                                step_level_validity_scores = scores["validity"],
                                step_level_redundancy_scores = scores["redundancy"],
                                step_level_validity_labels = [item > self.validity_threshold for item in scores["validity"]],
                                step_level_redundancy_labels = [item > self.redundancy_threshold for item in  scores["redundancy"]],
                            )
                            res = dict(scores=score_dict, idx=idx, validity=True)
                        else:
                            res = dict(validity=False, idx=idx, original_response=response)
                        dataloader.dataset.store_results(res)
                        # log = dict(idx = current_idx, response = current_response, scores = scores, result = res)
                        # dataloader.dataset.save_result_item_into_log(log,self.log_save_dir)
                    except:
                        current_response = response
                        logger.error(f"Error in responding to idx {idx}")
                        res = dict(validity=False, idx=idx, original_response=current_response)
                        dataloader.dataset.store_results(res)
                        # log = dict(idx = current_idx, response = current_response, scores = None, result = res)
                        # dataloader.dataset.save_result_item_into_log(log,self.log_save_dir)
                if progress_bar is not None:
                    progress_bar.update(self.save_to_ckpt_interval)
        