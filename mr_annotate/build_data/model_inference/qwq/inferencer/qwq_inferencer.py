import os
import json
import tqdm
import torch
import re
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from peft import PeftModel
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass, field

from typing import Dict, Sequence, Optional,List
from accelerate import PartialState,Accelerator
from tqdm import tqdm
from functools import partial
import threading
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer

from .qwq_inferencer_dataset import dataset_dict, DataCollatorQwQdDataset




class BaseQwenInferencer():
    
    def __init__(self,inference_args,data_args,model_args):
        
        self.inference_args, self.data_args, self.model_args = inference_args, data_args, model_args
        
        self.initialize_model()
        self.prepare_dataset()
        
        
    def prepare_dataset(self):
        raise NotImplementedError
    
    def initialize_model(self):
        raise NotImplementedError
        
    def inference(self):
        raise NotImplementedError


class QwQGeneratePRMInferencer(BaseQwenInferencer):
    def extract_steps(self,text):
        """
        从文本中提取每个 Step 的内容，并按顺序返回一个列表。
        """
        # 正则表达式：匹配 "Step X." 开头，捕获其后的内容
        pattern = r"(Step \d+\..*?)(?=Step \d+\.|\Z)"  # 匹配 Step 开头到下一个 Step 或文本结束
        steps = re.findall(pattern, text, re.DOTALL)  # 使用 re.DOTALL 允许匹配换行符
        return steps
    
    def prepare_dataset(self):
        self.function = self.inference_args.function
        self.dataset = dataset_dict[self.function](self.data_args)
    
    def initialize_model(self):
        self.model_name = self.model_args.model_path

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
    
    def inference(self):
        for idx, item in enumerate(tqdm(self.dataset)):
            messages = item["messages"]
            item_idx = item["idx"]
            question = item["question"]
            process_list = item["process_list"]
            ground_truth = item["ground_truth"]
            
            # print(messages)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            response_list = self.extract_steps(response)
            res_item = dict(
                original_question = question,
                modified_question = question,
                original_process = process_list,
                modified_process = response_list,
                modified_steps = [],
                error_steps = [],
                reason = "",
                ground_truth = ground_truth,
                idx=item_idx,
                question=question,
                classification="one_question_multi_answers",
                original_response = response
            )
            self.dataset.write_output_item(res_item)
            
class QwQParallelGeneratePRMInferencer(BaseQwenInferencer):
    def extract_steps(self,text):
        """
        从文本中提取每个 Step 的内容，并按顺序返回一个列表。
        """
        # 正则表达式：匹配 "Step X." 开头，捕获其后的内容
        pattern = r"(Step \d+\..*?)(?=Step \d+\.|\Z)"  # 匹配 Step 开头到下一个 Step 或文本结束
        steps = re.findall(pattern, text, re.DOTALL)  # 使用 re.DOTALL 允许匹配换行符
        return steps
    
    def prepare_dataset(self):
        self.function = self.inference_args.function
        self.dataset = dataset_dict[self.function](self.data_args)
        data_collator = DataCollatorQwQdDataset(self.tokenizer)
        self.dataloader = DataLoader(
            self.dataset,
            collate_fn=data_collator,
            batch_size=self.data_args.batch_size,
            num_workers=self.data_args.num_workers
        )
    
    def initialize_model(self):
        self.model_name = self.model_args.model_path

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        self.accelerator = Accelerator()
    
    def inference(self):
        self.model = self.model.to(self.accelerator.device)
        self.dataloader = self.accelerator.prepare(self.dataloader)
        
        for batch in tqdm(self.dataloader):
            with torch.no_grad():
                model_inputs = batch["model_inputs"].to(self.accelerator.device)
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=2048
                )
             
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for i in range(len(response)):
                    response_list = self.extract_steps(response[i])
                    res_item = dict(
                        original_question = batch["question"][i],
                        modified_question = batch["question"][i],
                        original_process = batch["process_list"][i],
                        modified_process = response_list,
                        modified_steps = [],
                        error_steps = [],
                        reason = "",
                        ground_truth = batch["ground_truth"][i],
                        idx=batch["idx"][i],
                        question=batch["question"][i],
                        classification="one_question_multi_answers",
                        original_response = response[i]
                    )
                    self.dataset.write_output_item(res_item)
                    
    

    
    