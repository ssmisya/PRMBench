from mr_eval.utils.task_utils import *
from mr_eval.utils.utils import *
from mr_eval.utils.log_utils import get_logger
from collections import Counter

import os
import re
import random
import math


logger = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "config.yaml"
file_path = os.path.join(current_dir, file_name)
task_config = load_task_config(file_path)
if "dataset_path" in task_config and os.path.isabs(task_config["dataset_path"]) == False:
    task_config["dataset_path"] = os.path.join(current_dir,task_config["dataset_path"])

def load_data_function():
    # Load Meta data
    raw_data = load_dir_of_jsonl_data_function_default(task_config)
    meta_data = []
    for idx, item in enumerate(raw_data):
        responses = item["responses"]
        question = item["question"]
        big_idx = item["id"]
        dataset = big_idx.split("_")[0].strip()
        ans_str = item["ans_str"]
        answer = item["answer"]
        labels = item["labels"]
        # answer = ans_str.split("####")[1].strip()
        n = len(responses)
        for response_id,response in enumerate(responses):
            raw_steps = response.split("\n\n")
            steps = []
            for raw_step in raw_steps:
                step = raw_step.strip()
                if step and len(step) > 0:
                    steps.append(step)
                    
            small_idx = f"{big_idx}/{response_id}"
            res = dict(
                idx = small_idx,
                question = question, 
                steps = steps, 
                step_str = response,
                dataset = dataset,
                ans_str = ans_str,
                answer = answer,
                n = n,
                label = labels[response_id],
            )
            meta_data.append(res)
            
    ## remove redundant items
    meta_dict = {}
    renewed_data = []
    for item in meta_data:
        idx = item["idx"]
        if meta_dict.get(idx) is None:
            meta_dict[idx] = 1
            renewed_data.append(item)
    meta_data = renewed_data
    
    ## Show statistics
    classification_dict = {}
    for item in meta_data:
        classification = item["dataset"]
        classification_dict[classification] = classification_dict.get(classification,0)+1 
        
    for k,v in classification_dict.items():
        logger.info(f"Dataset: {k}, number: {v}")
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results, meta_data):
    ## Prepare meta data
    small_id_dict = {meta["idx"]: meta for meta in meta_data}
    dataset_types = set([meta["dataset"] for meta in meta_data])
    big_id_dict = {}
    for meta in meta_data:
        big_id = meta["idx"].split("/")[0].strip()
        if big_id_dict.get(big_id) is None:
            big_id_dict[big_id] = []
        big_id_dict[big_id].append(meta)
    
    ## prepare results data
    filtered_dict = {}
    filtered_results = []
    for result in results:
        idx = result["idx"]
        if filtered_dict.get(idx) is None and small_id_dict.get(idx) is not None:
            filtered_dict[idx] = 1
            filtered_results.append(result)
    res_small_id_dict = {result["idx"]: result for result in filtered_results}
    res_big_id_dict = {}
    for result in filtered_results:
        big_id = result["idx"].split("/")[0].strip()
        if res_big_id_dict.get(big_id) is None:
            res_big_id_dict[big_id] = []
        res_big_id_dict[big_id].append(result)
        
    ## Evaluate
    pass_at_n = eval_pass_at_n(dataset_types, big_id_dict)
    maj_of_n = eval_maj_of_n(dataset_types, big_id_dict)
    prm_bon = eval_prm_bon(dataset_types, res_big_id_dict, big_id_dict, small_id_dict)
    
    return dict(
        pass_at_n = pass_at_n,
        maj_of_n = maj_of_n,
        prm_bon = prm_bon,
    )
    
def eval_pass_at_n(dataset_list, big_id_dict):
    res = {}
    for dataset_name in dataset_list:
        pass_num = 0
        total_num = 0
        for big_id, big_id_meta in big_id_dict.items():
            assert len(big_id_meta) > 0
            if len(big_id_meta) > 0 and big_id_meta[0]["dataset"] == dataset_name:
                total_num += 1
                for item in big_id_meta:
                    if item["label"] == "correct":
                        pass_num += 1
                        break
                    # if "labels" in item:
                    #     if "correct" in item["labels"]:
                    #         pass_num += 1
                    #         break
                    # else:
                    #     response_answer = extract_answer_from_model_response(item["step_str"])
                    #     if response_answer and response_answer and response_answer == item["answer"]:
                    #         pass_num += 1
                    #         break
        
        dataset_pass_at_n = pass_num / total_num if total_num > 0 else -1
        res[dataset_name] = dataset_pass_at_n
    return res

def eval_maj_of_n(dataset_list, big_id_dict):
    res = {}
    for dataset_name in dataset_list:
        pass_num = 0
        total_num = 0
        for big_id, big_id_meta in big_id_dict.items():
            assert len(big_id_meta) > 0
            if len(big_id_meta) > 0 and big_id_meta[0]["dataset"] == dataset_name:
                total_num += 1
                current_answer_dict = {}
                current_answer_list = []
                for item in big_id_meta:
                    response_answer = extract_answer_from_model_response(item["step_str"])
                    current_answer_list.append(response_answer)
                    
                counter = Counter(current_answer_list)
                max_count = max(counter.values())
                most_common = [item for item, freq in counter.items() if freq == max_count and item is not None]
                candidate_idxs = [idx for idx, item in enumerate(current_answer_list) if item in most_common]
                if len(candidate_idxs) > 0:
                    choice_idx = random.choice(candidate_idxs)
                    # most_chosen_one = current_answer_list[choice_idx]
                    most_chosen_item = big_id_meta[choice_idx]
                    if most_chosen_item["label"] == "correct":
                        pass_num += 1
                        
        dataset_maj_n = pass_num / total_num if total_num > 0 else -1
        res[dataset_name] = dataset_maj_n
    return res


def eval_prm_bon(dataset_list, res_big_id_dict, meta_big_id_dict, meta_small_id_dict):
    res = {"qualified_rate":{},"result":{}}
    for dataset_name in dataset_list:
        pass_num = 0
        total_num = 0
        not_qualify = 0
        for big_id, big_id_res in res_big_id_dict.items():
            big_id_meta = meta_big_id_dict.get(big_id,None)
            if big_id_meta is None or len(big_id_meta) == 0:
                logger.warning(f"big_id {big_id} not found in meta data")
                continue
            if big_id_meta[0]["dataset"] == dataset_name:
                total_num += 1
                
                ## Calculate completion score ： product of all step scores
                all_step_scores = []
                for item in big_id_res:
                    if "validity" in item and not item["validity"]:
                        continue
                    step_scores = item["scores"]["step_level_validity_scores"] if "step_level_validity_scores" in item["scores"] else []
                    all_step_scores.extend(step_scores)
                min_step_score = min(all_step_scores) - 0.01 if len(all_step_scores) > 0 else -0.01
                
                completion_score_info = []
                for item in big_id_res:
                    if "validity" in item and not item["validity"]:
                        continue
                    step_scores = item["scores"]["step_level_validity_scores"] if "step_level_validity_scores" in item["scores"] else []
                    step_scores_adjust = [score - min_step_score for score in step_scores]
                    completion_score = math.prod(step_scores_adjust)
                    completion_score_info.append({"idx":item["idx"],"completion_score":completion_score})
                completion_score_info = sorted(completion_score_info,key=lambda x:x["completion_score"],reverse=True)
                
                if len(completion_score_info) == 0:
                    not_qualify += 1
                    selected_item = random.choice(big_id_meta)
                    # continue
                else:
                    selected_item = completion_score_info[0]
                
                ## Find correctness of selected item
                selected_idx = selected_item["idx"]
                selected_meta = meta_small_id_dict.get(selected_idx)
                if selected_meta is None:
                    logger.warning(f"selected_idx {selected_idx} not found in meta data")
                else:
                    response_answer = extract_answer_from_model_response(selected_meta["step_str"])
                    # if response_answer and selected_meta["answer"] and response_answer == selected_meta["answer"]:
                    #     pass_num += 1
                    if selected_meta["label"] == "correct":
                        pass_num += 1
                
        dataset_qualify_rate = (total_num - not_qualify) / total_num if total_num > 0 else -1
        dataset_res = pass_num / total_num if total_num > 0 else -1
        res["result"][dataset_name] = dataset_res
        res["qualified_rate"][dataset_name] = dataset_qualify_rate
    return res
    
                    
def get_max_key(d):
    if len(d) == 0:
        return None
    max_value = max(d.values())
    max_keys = [key for key, value in d.items() if value == max_value]
    return random.choice(max_keys)

def extract_answer_from_model_response(model_response):
    # 使用正则表达式匹配 \boxed{内容}
    matches = re.findall(r'\\boxed\{(.*?)\}', model_response)
    if matches:
        return str(matches[-1])
    return None
                    
def get_most_common(lst):
    count = Counter(lst)
    
    max_count = max(count.values())
    
    most_common = [item for item, freq in count.items() if freq == max_count]

    return random.choice(most_common)    
    
 