import json

import re
from copy import deepcopy
import argparse
from openai import OpenAI
from prompts import prompt_dict

import os,sys,openai,time
from mr_eval.utils.utils import *

from copy import deepcopy
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)
def answer_sequence_to_str(answer_sequence):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"Step {idx+1}. {step['text']}\n\n")
    res_str = "".join(res)
    return res_str

def answer_sequence_to_shepherd_str(answer_sequence,step_tag = 'ки'):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"Step {idx+1}: {step['text']} {step_tag}\n")
    res_str = "".join(res)
    return res_str

def answer_sequence_to_reasoneval_list(answer_sequence):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"{idx+1}. {step['text']}")
    return res
    

def get_best_answer_by_item(item,return_type="shepherd"):
    steps = item["label"]["steps"]
    best_answers = []
    for step in steps:
        if step["human_completion"] is not None and step["chosen_completion"] is None:
            best_answers.append(step["human_completion"])
        elif step["chosen_completion"] is not None:
            best_answers.append(step["completions"][step["chosen_completion"]])
        else:
            logger.info(f"skipped one step")
    if return_type == "shepherd":
        answer_str = answer_sequence_to_shepherd_str(best_answers)
    elif return_type == "str":
        answer_str = answer_sequence_to_str(best_answers)
    elif return_type == "reasoneval":
        answer_str = answer_sequence_to_reasoneval_list(best_answers)
    else:
        answer_str =  best_answers
    return answer_str

def get_latex_str(question,answer):
    res = f"Question:\n\n{question}\n\nAnswer:\n\n{answer}"
    return res



def score_list_to_str(score_list):
    valid2_list = [str(round(i,2)) for i in score_list]
    res =  ", ".join(valid2_list)
    return res

def clean_str_inside_str(input_str):
    input_str = re.sub(r'([^\\])\\\\+([^\\\s])', r'\1\\\2', input_str)
    input_str = re.sub(r'([\s])\\([\s])', r'\1\\\\\2', input_str)
    input_str = re.sub(r'([\s])\\\\\\+([\s])', r'\1\\\\\2', input_str)
    return input_str

def clean_str(input_str):
    res_str = deepcopy(input_str)
    res_str = re.sub(r'\\+([^\\\s])', r'\\\\\1', res_str)
    res_str = re.sub(r'\\+([\s])', r'\\\\\\\\\1', res_str)
    return res_str


def json_str_to_object(json_str):
    if json_str.startswith("```json"):
        json_str = json_str.replace("```json", "").replace("```", "").strip()
    res_str = clean_str(json_str)
    try:
        json_object = json.loads(res_str)
        return json_object
    except:
        logger.error("Invalid JSON Str.")

def modify_file_name(file_name: str, add_str: str) -> str:
    base_name, ext = os.path.splitext(os.path.basename(file_name))
    new_name = f"{base_name}_{add_str}{ext}"
    return os.path.join(os.path.dirname(file_name), new_name)
        
def gpt4o_inference(
    input_data,
    processed_ids,
    output_file,
    failed_case_file,
    token = "",
    model_name = "gpt-4o",
    endpoint = "",
    classification = "all",
):
    
    client = OpenAI(
        base_url=endpoint,
        api_key=token,
    )
    for prm_item in tqdm(
            input_data, 
            total=len(input_data), 
            desc="Processing",
            mininterval=0,  
            miniters=1,
        ):
        idx = prm_item["idx"]
        if str(idx) in processed_ids:
            continue
        question = prm_item['question']["problem"]
        best_latex = get_best_answer_by_item(prm_item,return_type="str")
        latex_str = get_latex_str(question,best_latex)
        
        prompt = prompt_dict[classification]
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt["system"],
                    },
                    {
                        "role": "user",
                        "content": prompt["few_shot"][0][0],
                    },
                    {
                        "role": "assistant",
                        "content": prompt["few_shot"][0][1],
                    },
                    {
                        "role": "user",
                        "content": prompt["few_shot"][1][0],
                    },  
                    {
                        "role": "assistant",
                        "content": prompt["few_shot"][1][1],
                    },
                    {
                        "role": "user",
                        "content": latex_str,
                    }
                ],
                temperature=1.0,
                top_p=1.0,
                model=model_name,
            )
            resp = response.choices[0].message.content  
            json_object = json_str_to_object(resp)
            
            if json_object:
                json_object["idx"] = idx
                json_object["question"] = question
                json_object["classification"] = classification
                append_jsonl(json_object,output_file,)
            else:
                target_str = f"{idx}:{resp}"
                append_jsonl(target_str,failed_case_file)
        except Exception as e:
            logger.error(f"Failed to process {idx}.")
            logger.error(e)
            # target_str = f"{idx}:{latex_str}"
            time.sleep(1)
            continue

def load_data(input_file,output_file):
    res = process_jsonl(input_file)
    if os.path.exists(output_file):
        processed_ids = set([str(i["idx"]) for i in process_jsonl(output_file)])
        logger.info(f"Output file {output_file} exists, loading processed ids. Remaining {len(res) - len(processed_ids)} items.")
    else:
        processed_ids = set()
        os.makedirs(os.path.dirname(output_file),exist_ok=True)
        logger.info(f"Output file {output_file} does not exist, creating new file. Remaining {len(res)} items.")
    return res,processed_ids
    

def main():
    parser = argparse.ArgumentParser(description="Hallucination Data Generator.")

    # 添加参数
    parser.add_argument("--endpoint", type=str, help="The API endpoint URL.",default="https://api.datapipe.app/v1")
    parser.add_argument("--key", type=str, help="The API key for authentication.",default="")
    parser.add_argument("--model_name", type=str, help="Model name or ID.", default="gpt-4o")
    parser.add_argument("--input_file", type=str, help="input data file.", )
    parser.add_argument("--output_file",  type=str, help="output data file.")
    parser.add_argument("--failed_case_file",  type=str, help="failed case file.")
    parser.add_argument("--classes", type=str, help="The classes of the data.", default="all")
    parser.add_argument("--log_file", type=str, help="The log file.", default="generation.log")
    parser.add_argument("--log_level", type=str, help="The log level.", default="INFO")
    args = parser.parse_args()
    
    logger_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=logger_level,  # 设置日志级别
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
        datefmt="%Y-%m-%d %H:%M",  # 设置时间格式
        filename=args.log_file,  # 设置输出文件名
        filemode="w",  # 设置输出模式
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logger_level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    file_handler = logging.FileHandler(args.log_file, mode="w")
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # 清空默认的 handlers 并重新添加
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    if args.classes == "all":
        classes = ["circular", "confidence", "counterfactual", "step_contradiction", "domain_inconsistency", "redundency", "missing_condition", "deception"]
    else:
        classes = args.classes.strip().split(",")
        
    output_file = args.output_file
    failed_case_file = args.failed_case_file
    
    
    
    for classfication in classes:
        
        logger.info(f"Generating {classfication}.")
        output_file  = modify_file_name(args.output_file,classfication)
        failed_case_file = modify_file_name(args.failed_case_file,classfication)
        
        input_data,processed_ids = load_data(args.input_file,output_file)
        gpt4o_inference(
            input_data,
            processed_ids,
            output_file,
            failed_case_file,
            args.key,
            args.model_name,
            args.endpoint,
            classification = classfication,
        )
        logger.info(f"{classfication} generation finished.")


if __name__ == "__main__":
    main()