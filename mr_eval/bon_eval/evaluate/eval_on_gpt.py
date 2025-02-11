import argparse
from datasets import Dataset
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from mr_eval.utils.utils import *
from copy import deepcopy

evaluate_assistant_prompt_reachqa = """
You are an AI assistant tasked with evaluating the accuracy of generated responses based on a given ground truth statement and gound truth label. Please carefully analyze the question, the generated response, the statement and the label provided. Your task is to determine whether the response aligns with the label and explain your reasoning. Please follow these steps:

- Extract the relevant information from the ground truth statement, especially the core fact or conclusion.
- Compare the generated response with the extracted information and the ground truth label.
- Determine whether the generated response is correct or incorrect based on the alignment with the ground truth label and statement.

Your output should strictly follow this format:

- Thought: Describe your reasoning step by step, including the process of extracting information from the label and comparing it to the generated response.
- Answer: State either "The answer is correct" or "The answer is wrong."

Example:
Question: Based on the observed trends in ocean current intensities over the decades, determine in which decade two of the currents have exactly the same intensity.
Generated Response: The decade in which two of the currents have exactly the same intensity is the 1980s.
GT Statement: Examine the data for ocean current intensities over each decade. In 1980, the Kuroshio Current and Antarctic Circumpolar Current both have an intensity of 22 units. Therefore, the decade when these two currents have exactly the same intensity is 1980.
GT Label: 1980
Response:
Thought: The label indicates the decade is the 1980s, based on the year 1980. The generated response also states "1980s," which matches the label. 
Answer: The answer is correct.

Question: Between the decades 1970 and 2000, which ocean current exhibited the largest increase in intensity, and what was the magnitude of this increase?
Generated Response: The Gulf Stream exhibited the largest increase in intensity, with a magnitude of 20 arbitrary units.
GT Statement: Between 1970 and 2000, the Gulf Stream, Kuroshio Current, and Antarctic Circumpolar Current each exhibited an increase in intensity of 25 units. Thus, they all had the same magnitude of increase.
GT Label: 25 units.
Response:
Thought: The label states that between 1970 and 2000, all three ocean currents (Gulf Stream, Kuroshio Current, and Antarctic Circumpolar Current) exhibited the same magnitude of increase in intensity, which is 25 units. The generated response incorrectly claims that the Gulf Stream exhibited the largest increase with a magnitude of 20 units, which contradicts the label.
Answer: The answer is wrong.
"""

class BenchmarkEvaluator:
    
    def __init__(self, args):
        self.args = args
        self.input_path = args.input_path
        self.output_path = args.output_path
        self.ckpt_path = args.ckpt_path
        self.model_path = args.model_path
        self.tensor_parallel_size = args.tensor_parallel_size
        self.task = args.task
        self.save_interval = args.save_interval
        
        self.load_prompt()
        self.load_data()
        self.load_model()
  
        self.resume_from_checkpoint()
    
    def load_model(self):
        self.model = LLM(
            model=self.model_path, 
            tensor_parallel_size = self.tensor_parallel_size, 
            trust_remote_code=True,
            # cpu_offload_gb=self.cpu_offload_gb
        )
        
        self.sampling_params = SamplingParams(
            temperature=0,
            top_k=-1,
            top_p=1,
            max_tokens = 2048
        )
        
    def load_prompt(self):
        self.prompt = evaluate_assistant_prompt_reachqa
        self.messages = [
            {"role": "user", "content": self.prompt}
        ]
        
    
    def load_data(self,):
        self.load_data_infereced_data()
    

    def load_data_infereced_data(self,):
        raw_data = process_jsonl(self.input_path)
        self.raw_data = raw_data
        self.meta_data = []
        self.conversations = []
        for data in raw_data:
            question = data['question']
            ans_str = data['ans_str']
            answer = data['answer']
            idx = data['id']
            respoonses = data['responses']
            for response_idx, response in enumerate(respoonses):
                small_id = f"{idx}/{response_idx}"
                conversation = f"""
                    Question: {question}
                    Generated Response: {response}
                    GT Statement: {ans_str}
                    GT Label: {answer}
                    Response:
                """
                data_instance = dict(
                    small_id = small_id,
                    question = question,
                    ans_str = ans_str,
                    answer = answer,
                    response = response,
                )
                messages = deepcopy(self.messages)
                messages.append({"role": "user", "content": conversation})
                self.conversations.append(messages)
                self.meta_data.append(data_instance)
        self.origin_data = self.meta_data
        
    def resume_from_checkpoint(self):
        if os.path.exists(self.ckpt_path):
            ckpt = process_jsonl(self.ckpt_path)
            processed_ids = {item["small_id"] for item in ckpt}
            renewed_data = []
            renewed_conversations = []
            for item,conv in zip(self.meta_data, self.conversations):
                if item["small_id"] not in processed_ids:
                    renewed_data.append(item)
                    renewed_conversations.append(conv)
            self.meta_data = renewed_data
            self.conversations = renewed_conversations
            print(f"Resuming from checkpoint, {len(self.meta_data)} items left.")
            
    def generate(self):
        self.results = []
        for i in range(0, len(self.conversations), self.save_interval):
            end_loc = min(i+self.save_interval, len(self.conversations))
            current_conversations = self.conversations[i:end_loc]
            outputs = self.model.chat(current_conversations, sampling_params=self.sampling_params)
            
            for j in range(i, end_loc):
                current_output = outputs[j-i].outputs[0].text
                original_item = self.meta_data[j]
                small_id = original_item["small_id"]
                original_conversation = self.conversations[j]
                user_prompt = original_conversation[-1]["content"]
                
                result = dict(
                    small_id = small_id,
                    question = original_item["question"],
                    prompt = user_prompt,
                    eval_model_response = current_output,
                )

                append_jsonl(result, self.ckpt_path)
            
    def evaluate(self):
        ckpt_data = process_jsonl(self.ckpt_path)
        assert len(ckpt_data) == len(self.origin_data)
        # Extract label
        small_id_dict = {}
        for item in ckpt_data:
            small_id = item["small_id"]
            eval_model_response = item["eval_model_response"]
            extracted_response = eval_model_response.lower().split("answer:")[-1].strip()
            if extracted_response and len(extracted_response) < 50:
                if extracted_response == "the answer is correct.":
                    label = "correct"
                elif extracted_response == "the answer is wrong.":
                    label = "wrong"
                else:
                    label = "unknown"
            else:
                label = "unknown"
                
            item["label"] = label
            small_id_dict[small_id] = item
        
        ## apply to the original data
        for item in self.raw_data:
            idx = item["id"]
            item["labels"] = []
            for i in range(len(item["responses"])):
                small_id = f"{idx}/{i}"
                
                if small_id in small_id_dict:
                    assert small_id_dict[small_id]["question"] == item["question"]
                    item["labels"].append(small_id_dict[small_id]["label"])
                else:
                    item["labels"].append("unknown")
        write_jsonl(self.raw_data, self.output_path)
        
        

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on the dataset.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True,)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True,)
    parser.add_argument("--tensor_parallel_size", type=int, default=8,)
    parser.add_argument("--task", type=str, default="bon")
    parser.add_argument("--save_interval", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()  # Parse arguments from command line
    evaluator = BenchmarkEvaluator(args)
    evaluator.generate()
    evaluator.evaluate()
    
    