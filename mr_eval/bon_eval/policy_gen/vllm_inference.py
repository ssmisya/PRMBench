
from vllm import LLM, SamplingParams
from datasets import load_dataset
from copy import deepcopy


from mr_eval.utils.utils import *
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union, Optional
from transformers import HfArgumentParser


@dataclass
class DataArguments:

    input_path: str = field(default=None)
    output_path: str = field(default=None)
    eval_maj_n: Optional[bool] = field(default=False)
    pretrained: str = field(default=None)
    tensor_parallel_size: int = field(default=8)
    temperature: float = field(default=0.7)
    top_k: int = field(default=-1)
    top_p: float = field(default=1.0)
    num_return_sequences: int = field(default=8)
    task: str = field(default="gsm8k")
    save_interval: int = field(default=100)
    cpu_offload_gb: int = field(default=0)
    

class VllmInference():
    def __init__(
        self,
        args,
    ):
        self.pretrained = args.pretrained
        self.tensor_parallel_size = args.tensor_parallel_size
        
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.num_return_sequences = args.num_return_sequences
        
        self.task = args.task
        self.input_path = args.input_path
        self.output_path = args.output_path
        
        self.eval_maj_n = args.eval_maj_n
        self.save_interval = args.save_interval
        self.cpu_offload_gb = args.cpu_offload_gb
        
        
        self.load_model()
        self.load_prompt()
        self.load_data()
        self.resume_from_checkpoint()
        
    def load_model(self):
        self.model = LLM(
            model=self.pretrained, 
            tensor_parallel_size = self.tensor_parallel_size, 
            trust_remote_code=True,
            cpu_offload_gb=self.cpu_offload_gb
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            n=self.num_return_sequences,
            max_tokens = 2048
        )
    
    def load_data(self):
        if self.task in ["gsm8k"]:
            self.load_data_hf(self.input_path)
        else:
            raise NotImplementedError("Task not implemented")
        
        
        
    def load_data_hf(self, data_path):
        dataset = load_dataset(data_path,"main")["test"]
        
        self.data = []
        self.conversations = []
        for idx,item in enumerate(dataset):
            # Load meta data
            question = item["question"]
            answer = item["answer"]
            item_idx = f"{self.task}_{idx}"
            res = {"id":item_idx,"question":question,"answer":answer}
            self.data.append(res)
            
            # Load conversation
            messages = deepcopy(self.messages)
            messages.append({"role": "user", "content": question})
            self.conversations.append(messages)
            
            
    
    def resume_from_checkpoint(self):
        if os.path.exists(self.output_path):
            ckpt = process_jsonl(self.output_path)
            processed_ids = {item["id"] for item in ckpt}
            renewed_data = []
            renewed_conversations = []
            for item,conv in zip(self.data,self.conversations):
                if item["id"] not in processed_ids:
                    renewed_data.append(item)
                    renewed_conversations.append(conv)
            self.data = renewed_data
            self.conversations = renewed_conversations
            print(f"Resuming from checkpoint, {len(self.data)} items left.")
            

    def load_prompt(self):
        self.prompt = "Please reason step by step, and put your final answer within \boxed{}."
        self.messages = [
            {"role": "user", "content": self.prompt}
        ]

    def generate(self):
        
        conversation_iter = iter(self.conversations)
        self.results = []
        for i in range(0, len(self.conversations), self.save_interval):
            end_loc = min(i+self.save_interval, len(self.conversations))
            current_conversations = self.conversations[i:end_loc]
            outputs = self.model.chat(current_conversations, sampling_params=self.sampling_params)
            
            for j in range(i, end_loc):
                current_outputs = outputs[j-i].outputs
                responses = [item.text for item in current_outputs]
                original_item = self.data[j]
                answer = original_item["answer"]
                question = original_item["question"]
                idx = original_item["id"]
                res = {"id":idx,"question":question,"answer":answer,"responses":responses}
                if self.eval_maj_n:
                    maj_n = self.eval_maj_n(responses, answer)
                    res["maj_n"] = maj_n

                append_jsonl(res, self.output_path)
        
    def eval_maj_n(self, responses, answers):
        pass
        
        
if __name__ == "__main__":
    qwq = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/QwQ-32B-Preview"
    r1 = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/DeepSeek-R1"
    r1_zero = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/DeepSeek-R1-Zero"
    r1_distil_qwen = "/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/policy_models/DeepSeek-R1-Distill-Qwen-7B"
    
    parser = HfArgumentParser((DataArguments,))
    inference_args = parser.parse_args_into_dataclasses()[0]
    
    inference_module = VllmInference(inference_args)
    inference_module.generate()