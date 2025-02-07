import os
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm
from accelerate import Accelerator
from openai import OpenAI
from datetime import datetime

from .abstract_model import prm
from ..utils.utils import *
from ..utils.prompts import PROMPT_DICT
from ..utils.model_utils import process_policy_lm_evaluation_response, remove_step_prefix
import time
import os
from ..utils.log_utils import get_logger 

logger = get_logger(__name__)

class OpenaiModels(prm):
    def __init__(
            self,
            model_name = "gpt-4o",
            endpoint = "https://api.datapipe.app/v1",
            api_key = os.environ.get("OPENAI_API_KEY",""),
            max_retry = 5,
            retry_interval = 5,
            # log_save_dir = "./mr_eval/scripts/logs/generated/openai_models.jsonl",
            validity_threshold = 0,
            redundancy_threshold = 0,
            shots=2,
        ) -> None:
        super().__init__(
            validity_threshold=validity_threshold,
            redundancy_threshold=redundancy_threshold,
        )
        
        self.model_name = model_name
        if self.model_name in ["o1-mini","o1-preview"]:
            self.first_round_role = "user"
            self.default_tempreture = 1.0
        else:
            self.first_round_role = "system"
            self.default_tempreture = 0.0
        
        self.shots = int(shots)
        self.endpoint = endpoint
        self.api_key = api_key
        self.max_retry = max_retry
        self.retry_interval = retry_interval
        self.tokenizer = None
        # self.log_save_dir = log_save_dir
        # current_time = datetime.now()
        # file_name_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        # self.log_save_dir = f"{log_save_dir[:-5]}_{file_name_time}.jsonl"
        
        self.prompt = PROMPT_DICT["policy_model_as_an_evaluator"]
        
        self.client = OpenAI(
                            base_url=self.endpoint,
                            api_key=self.api_key,
                        )
        
        self.accelerator = Accelerator(
            cpu=True,  # 可选：强制使用 CPU
            split_batches=False,  # 禁止批次分割
        )
        logger.info(f"Few shot setting: {self.shots}")

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
        
        
        res = dict(
            idx = data_idx,
            inputs = original_input_for_prm,
            model_type = "openai",
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:
        
        if self.accelerator.is_main_process:
            for batch in tqdm(dataloader, desc="Model Responding"):
                idx = batch['idx']
                inputs = batch['inputs']
                fail_times = 0
                for i in range(self.max_retry):
                    fail_flag = False
                    gpt_message = [
                        {
                            "role": self.first_round_role,
                            "content": self.prompt["system_prompt"],
                        },
                    ]
                    assert self.shots < 3, "Few-shot evaluation only supports 1 or 2 shots"
                    for shot in range(self.shots):
                        gpt_message.append(
                            {
                                "role": "user",
                                "content": self.prompt["fewshots"][shot][0],
                            }
                        )
                        gpt_message.append(
                            {
                                "role": "assistant",
                                "content": self.prompt["fewshots"][shot][1],
                            }
                        )
                    gpt_message.append(
                        {
                            "role": "user",
                            "content": inputs,
                        }
                    )
                    try:
                        # ## Debug
                        # import debugpy
                        # debugpy.listen(address = ('0.0.0.0', 7119))
                        # debugpy.wait_for_client() 
                        # breakpoint() # 在下一句代码处暂停
                        # # dist.barrier()
                        
                        response = self.client.chat.completions.create(
                                messages=gpt_message,
                                temperature=self.default_tempreture,
                                model=self.model_name,
                        )

                        response = response.choices[0].message.content  
                        scores = process_policy_lm_evaluation_response(response)
                        if scores:
                            score_dict = dict(
                                step_level_validity_scores = scores["validity"],
                                step_level_redundancy_scores = scores["redundancy"],
                                step_level_validity_labels = [item > 0 for item in scores["validity"]],
                                step_level_redundancy_labels = [item > 0 for item in  scores["redundancy"]],
                            )
                            res = dict(scores=score_dict, idx=idx, validity=True)
                        else:
                            res = dict(validity=False, idx=idx, original_response = response)
                        dataloader.dataset.store_results(res)
                        log = dict(idx = idx, inputs = inputs, response = response, scores = scores, result = res)
                        # dataloader.dataset.save_result_item_into_log(log,self.log_save_dir)
                        break
                    except Exception as e:
                        eval_logger.error(f"Error: {e}, retrying in {self.retry_interval + fail_times*self.retry_interval} seconds")
                        fail_times += 1
                        fail_flag = True
                        time.sleep(self.retry_interval + fail_times*self.retry_interval)
                    except AssertionError as e:
                        eval_logger.error(f"Error: {e}, retrying in {self.retry_interval + fail_times*self.retry_interval} seconds")
                        fail_times += 1
                        fail_flag = True
                        time.sleep(self.retry_interval + fail_times*self.retry_interval)

                if i == self.max_retry - 1 and fail_flag:
                    eval_logger.error(f"Failed to get response for idx: {idx}")
                    res = dict(validity=False, idx=idx)
                    dataloader.dataset.store_results(res)
                    try:
                        log = dict(idx = idx, inputs = inputs, response = response, scores = None, result = res)
                        # dataloader.dataset.save_result_item_into_log(log,self.log_save_dir)
                    except:
                        pass
                    
        self.accelerator.wait_for_everyone()
        
        
   