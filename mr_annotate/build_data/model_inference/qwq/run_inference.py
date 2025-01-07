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

from inferencer import inferencer_type_dict

if __name__ == "__main__":

    @dataclass
    class DataArguments:

        input_path: List[str] = field(default_factory=list)
        output_path: str = field(default=None)
        batch_size: int = field(default=8)
        num_workers: int = field(default=2)
        
    @dataclass
    class ModelArguments:
        model_path: str = field(default="/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/QwQ-32B-Preview")

        
    @dataclass
    class InferenceArguments:
        function: str = field(default="generate_prm")
        parallel_mode: str = field(default="1gpu")
        float_type: str = field(default="float16")
        def __post_init__(self):
            self.float_type = getattr(torch, self.float_type)
            
    parser = HfArgumentParser(
    (InferenceArguments, ModelArguments, DataArguments))
    
    
    inference_args, model_args, data_args = parser.parse_args_into_dataclasses()
    data_args.input_path =  data_args.input_path[0]
    function_name = inference_args.function
    inference_module = inferencer_type_dict[function_name]["model"](inference_args=inference_args,model_args=model_args,data_args=data_args)

    inference_module.inference()