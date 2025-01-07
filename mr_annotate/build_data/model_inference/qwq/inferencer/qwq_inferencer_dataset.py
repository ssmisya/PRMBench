from torch.utils.data import DataLoader, Dataset
from mr_eval.utils.utils import *
from copy import deepcopy
from typing import Dict, Sequence, Optional, List
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import torch

class QwQBaseDataset(Dataset):
    def __init__(self, data_args):
        self.data_args = data_args
        self.input_path = data_args.input_path
        self.output_path = data_args.output_path
        

        self.set_prompt()
        self.resume_from_ckpt()
        self.load_data()
    
    def __len__(self):
        raise NotImplementedError
    
    def load_data(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def set_prompt(self):
        raise NotImplementedError
    
    def resume_from_ckpt(self):
        raise NotImplementedError
    
    def write_output_item(self,item):
        raise NotImplementedError
    

class QwQGeneratePRMDataset(QwQBaseDataset):
    
    def resume_from_ckpt(self):
        if os.path.exists(self.output_path):
            print_rank0(f"Ckpt exists, Loading from {self.output_path}")
            self.cache = process_jsonl(self.output_path)
            self.processd_idx = {item["idx"]:1 for item in self.cache}
        else:
            print_rank0(f"Ckpt not exists, Creating from scratch")
            self.cache = []
            self.processd_idx = {}
    
    def load_data(self):
        raw_data = process_jsonl(self.input_path)
        self.meta_data = []
        for item in raw_data:
            idx = item["idx"]
            if idx in self.processd_idx:
                continue
            question = item['question']["problem"]
            ground_truth = item['question']["ground_truth_answer"]
            steps = item["label"]["steps"]
            
            process_list, process_str = get_best_answer_by_item(item)
            messages = deepcopy(self.messages)
            messages.append({"role": "user", "content": question})
            self.meta_data.append(dict(idx=idx, messages=messages, process_list=process_list,question=question,ground_truth=ground_truth))
        
    def write_output_item(self, item):
        append_jsonl(item, self.output_path)
    
    def set_prompt(self):
        fewshot_q1 ="Compute $\\arcsin \left( -\\frac{1}{2} \\right).$  Express your answer in radians."
        fewshot_a1="""
        Step 1. I know that the arcsine function is the inverse of the sine function, so I want to find an angle $\\theta$ such that $\sin(\\theta) = -\\frac{1}{2}.$

        Step 2. I also know that the range of the arcsine function is $[-\\frac{\pi}{2}, \\frac{\pi}{2}]$, so I only need to consider angles in the fourth and first quadrants, where sine is negative and positive respectively.

        Step 3. I recall that the sine function is periodic with a period of $2\pi$, so any angle that satisfies $\sin(\\theta) = -\\frac{1}{2}$ must be of the form $\\theta = -\\frac{\pi}{6} + 2k\pi$ or $\\theta = \\frac{7\pi}{6} + 2k\pi$, where $k$ is an integer.

        Step 4. However, since I want $\\theta$ to be in the range of the arcsine function, I need to choose $k$ such that $-\\frac{\pi}{2} \leq \\theta \leq \\frac{\pi}{2}.$

        Step 5. This means that $k$ can only be 0 or -1, and the only possible values of $\\theta$ are $-\\frac{\pi}{6}$ or $\\frac{7\pi}{6}.$

        Step 6. To decide which one is the correct answer, I can use the fact that the arcsine function is an odd function, meaning that $\\arcsin(-x) = -\\arcsin(x)$ for any $x$ in the domain.

        Step 7. Therefore, since I have $\\arcsin \left( -\\frac{1}{2} \\right)$, I need to take the negative of the angle that gives $\sin(\\theta) = \\frac{1}{2}$, which is $\\frac{\pi}{6}.$

        Step 8. So, the final answer is $\\theta = -\\frac{\pi}{6}.$

        # Answer

        -\\frac{\pi}{6}
        """

        fewshot_q2="If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."

        fewshot_a2="""
        Step 1. To solve an equation involving absolute value, I need to consider two cases: one where the expression inside the absolute value is positive, and one where it is negative.

        Step 2. For the first case, I assume that both $x+5$ and $3x-6$ are positive, so I can drop the absolute value signs and get $x+5=3x-6$.

        Step 3. Solving for $x$ in this case, I subtract $x$ from both sides and add $6$ to both sides, and get $11=2x$.

        Step 4. Dividing both sides by $2$, I get $x=\\frac{11}{2}$.

        Step 5. For the second case, I assume that both $x+5$ and $3x-6$ are negative, so I can change the signs of both expressions and get $-x-5=-3x+6$.

        Step 6. Solving for $x$ in this case, I add $3x$ to both sides and subtract $6$ from both sides, and get $-11=2x$.

        Step 7. Dividing both sides by $2$, I get $x=-\\frac{11}{2}$.

        Step 8. Now I have two possible values for $x$, but the problem asks for the largest one, so I compare them and see that $\\frac{11}{2}$ is larger than $-\\frac{11}{2}$.

        Step 9. Therefore, the largest possible value of $x$ that satisfies the equation is $\\frac{11}{2}$.

        # Answer

        \\frac{11}{2}
        """
        self.messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. And return answers as the following format: Step 1. xxx \n Step 2. xxx \n ...... Step n. xxx \n "},
            {"role": "user", "content": fewshot_q1},
            {"role": "assistant", "content": fewshot_a1},
            {"role": "user", "content": fewshot_q2},
            {"role": "assistant", "content": fewshot_a2},
            # {"role": "user", "content": prompt},
        ]
                
            
    def __len__(self):
        return len(self.meta_data)    
    
    def __getitem__(self, idx):
        return self.meta_data[idx]
    


def answer_sequence_to_str(answer_sequence):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"Step {idx+1}. {step['text']}\n\n")
    res_str = "".join(res)
    return res_str
def get_best_answer_by_item(prm_item):
    steps = prm_item["label"]["steps"]
    best_answers = []
    for step in steps:
        if step["human_completion"] is not None and step["chosen_completion"] is None:
            best_answers.append(step["human_completion"])
        elif step["chosen_completion"] is not None:
            best_answers.append(step["completions"][step["chosen_completion"]])
        else:
            print_rank0(f"skipped one step")
    process_list = [step["text"] for step in best_answers]
    answer_str = answer_sequence_to_str(best_answers)
    return process_list,answer_str


dataset_dict= dict(
    generate_prm=QwQGeneratePRMDataset,
    parallel_generate_prm=QwQGeneratePRMDataset,
)

@dataclass
class DataCollatorQwQdDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        messages = [instance["messages"] for instance in instances]
        texts = [ self.tokenizer.apply_chat_template(
            message_item,
            tokenize=False,
            add_generation_prompt=True
        ) for message_item in messages]
        model_inputs = self.tokenizer(texts, return_tensors="pt",padding=True)
        
        batch = dict(model_inputs = model_inputs)
        for k,v in instances[0].items():
            if k not in ["messages"]:
                batch[k] = [instance[k] for instance in instances]
        return batch
    