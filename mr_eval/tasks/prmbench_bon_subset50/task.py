from mr_eval.utils.task_utils import *
from mr_eval.utils.utils import *
from mr_eval.utils.log_utils import get_logger
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
                    response_answer = extract_answer_from_model_response(item["step_str"])
                    if response_answer and response_answer and response_answer == item["answer"]:
                        pass_num += 1
                        break
        
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
                for item in big_id_meta:
                    response_answer = extract_answer_from_model_response(item["step_str"])
                    if response_answer is not None:
                        current_answer_dict[response_answer] = current_answer_dict.get(response_answer,0)+1
                most_chosen_one = get_max_key(current_answer_dict)
                answer = big_id_meta[0]["answer"]
                if most_chosen_one and answer and most_chosen_one == answer:
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
                    if response_answer and selected_meta["answer"] and response_answer == selected_meta["answer"]:
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
                    
                    
            
    
    
#     meta_data_dict = {meta["idx"]: meta for meta in meta_data}
#     classification_types = set([meta["classification"] for meta in meta_data])
#     metric_types = ["correct_step_acc","wrong_step_acc","total_step_acc","first_error_acc"]
#     halucination_specified_dict = {}
#     total_metric_lists = {}
#     for metric in metric_types+["similarity"]:
#         halucination_specified_dict[metric] = {i:[] for i in classification_types}
#         total_metric_lists[metric] = []
#     halucination_specified_dict["f1_matrix"] = {i:dict(TP=0,FP=0,TN=0,FN=0) for i in classification_types}
#     total_metric_lists["f1_matrix"] = dict(TP=0,FP=0,TN=0,FN=0)
    

#     detailed_logs = []
#     valid_num = 0
#     total_num = len(meta_data)
    
#     ## Filter out repeated items
#     filtered_dict = {}
#     filtered_results = []
#     for result in results:
#         idx = result["idx"]
#         if filtered_dict.get(idx) is None and meta_data_dict.get(idx) is not None:
#             filtered_dict[idx] = 1
#             filtered_results.append(result)
    
#     assert abs(len(filtered_results) - len(meta_data)) < 5, f"filtered_results number: {len(filtered_results)}, meta_data number: {len(meta_data)}"

#     correct_ids_dict = {meta["idx"]:1 for meta in meta_data if meta["classification"] == "correct"} 
#     correct_predictions  = [prediction for prediction in filtered_results if prediction["idx"] in correct_ids_dict]
#     other_predictions = [prediction for prediction in filtered_results if prediction["idx"] not in correct_ids_dict]
#     correct_model_response_acc_dict = {}
    
#     ## First evaluate the correct samples
#     for prediction in correct_predictions:
#         idx = prediction["idx"]
#         reference_item = meta_data_dict[idx]
#         error_steps = reference_item["error_steps"]    
#         classifcation = reference_item["classification"]
#         assert classifcation == "correct"
        
#         if "validity" in prediction and not prediction["validity"]:
#             log = dict(
#                 idx=idx,
#                 error_steps=error_steps,
#                 classifcation=classifcation,
#                 prediction=None,
#                 results=None,
#                 )
#         else:
#             labels = prediction["scores"]["step_level_validity_labels"] if "step_level_validity_labels" in prediction["scores"] else None
#             res_dict = eval_on_hallucination_step(error_steps,labels)
                
#             for metric in metric_types:
#                 # total_metric_lists[metric].extend(res_dict[f'{metric}_list'])
#                 halucination_specified_dict[metric][classifcation].extend(res_dict[f'{metric}_list'])
#             halucination_specified_dict["f1_matrix"][classifcation]["TP"] += res_dict["f1_matrix"]["TP"]
#             halucination_specified_dict["f1_matrix"][classifcation]["FP"] += res_dict["f1_matrix"]["FP"]
#             halucination_specified_dict["f1_matrix"][classifcation]["TN"] += res_dict["f1_matrix"]["TN"]
#             halucination_specified_dict["f1_matrix"][classifcation]["FN"] += res_dict["f1_matrix"]["FN"]
            
#             correct_model_response_acc_dict[idx] = res_dict["model_response_acc"]   
#             log = dict(
#                 idx=idx,
#                 error_steps=error_steps,
#                 classifcation=classifcation,
#                 prediction=prediction,
#                 results=res_dict,
#             )
#             detailed_logs.append(log)
    
#     ## Then evaluate the other sample types
#     for prediction in other_predictions:
#         idx = prediction["idx"]
        
#         if "validity" in prediction and not prediction["validity"]:
#             log = dict(
#                 idx=idx,
#                 hallucination_steps=None,
#                 hallucination_types=None,
#                 prediction=None,
#                 results=None,
#                 validitiy=False,
#                 )
#         else:
#             valid_num += 1
#             try:
#                 reference_item = meta_data_dict[idx]
#             except:
#                 logger.info(f"idx {idx} not found in meta_data_dict")
#                 continue
#             error_steps = reference_item["error_steps"]
#             classifcation = reference_item["classification"]
            
#             if (classifcation == "redundency" or classifcation == "circular") and "step_level_redundancy_labels" in prediction["scores"]:
#                 labels = prediction["scores"]["step_level_redundancy_labels"]
#                 labels = [ not i for i in labels]
#                 res_dict = eval_on_hallucination_step(error_steps,labels,redundency_label=False)
#             else:
#                 labels = prediction["scores"]["step_level_validity_labels"] if "step_level_validity_labels" in prediction["scores"] else None
#                 res_dict = eval_on_hallucination_step(error_steps,labels,redundency_label=False)
                
#             for metric in metric_types:
#                 total_metric_lists[metric].extend(res_dict[f'{metric}_list'])
#                 halucination_specified_dict[metric][classifcation].extend(res_dict[f'{metric}_list'])
#             halucination_specified_dict["f1_matrix"][classifcation]["TP"] += res_dict["f1_matrix"]["TP"]
#             halucination_specified_dict["f1_matrix"][classifcation]["FP"] += res_dict["f1_matrix"]["FP"]
#             halucination_specified_dict["f1_matrix"][classifcation]["TN"] += res_dict["f1_matrix"]["TN"]
#             halucination_specified_dict["f1_matrix"][classifcation]["FN"] += res_dict["f1_matrix"]["FN"]
            
#             total_metric_lists["f1_matrix"]["TP"] += res_dict["f1_matrix"]["TP"]
#             total_metric_lists["f1_matrix"]["FP"] += res_dict["f1_matrix"]["FP"]
#             total_metric_lists["f1_matrix"]["TN"] += res_dict["f1_matrix"]["TN"]
#             total_metric_lists["f1_matrix"]["FN"] += res_dict["f1_matrix"]["FN"]
            
#             correct_idx = "correct_"+idx[len(f"{classifcation}_"):]
#             correct_item_acc = correct_model_response_acc_dict.get(correct_idx)
#             item_acc = res_dict["model_response_acc"]
#             if correct_item_acc and item_acc != -1:
#                 abs_similarity = abs(item_acc - correct_item_acc)  
#                 total_metric_lists["similarity"].append(abs_similarity)
#                 halucination_specified_dict["similarity"][classifcation].append(abs_similarity)
#             log = dict(
#                         idx=idx,
#                         error_steps=error_steps,
#                         classifcation=classifcation,
#                         prediction=prediction,
#                         results=res_dict,
#                     )
#         detailed_logs.append(log)
    
    
#     ## Calculate final results
#     total_final_results = {metric:sum(total_metric_lists[metric])/len(total_metric_lists[metric]) if len(total_metric_lists[metric])>0 else -1 for metric in metric_types+["similarity"]}
#     hallucination_type_final_results = {metric:{k:sum(v)/len(v) if len(v)>0 else -1 for k,v in halucination_specified_dict[metric].items()} for metric in metric_types+["similarity"]}
#     validitiy_rate = valid_num / total_num
    
    
#     ## Calculate F1 score
#     TP = total_metric_lists["f1_matrix"]["TP"]
#     FP = total_metric_lists["f1_matrix"]["FP"]
#     FN = total_metric_lists["f1_matrix"]["FN"]
#     TN = total_metric_lists["f1_matrix"]["TN"]
#     total_precision = TP / (TP + FP) if (TP + FP) != 0 else -1
#     total_recall = TP / (TP + FN) if (TP + FN) != 0 else -1
#     total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) != 0 else -1
#     negative_precision = TN / (TN + FN) if (TN + FN) != 0 else -1
#     negative_recall = TN / (TN + FP) if (TN + FP) != 0 else -1
#     negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if (negative_precision + negative_recall) != 0 else -1
#     total_final_results["precision"] = total_precision
#     total_final_results["recall"] = total_recall
#     total_final_results["f1"] = total_f1
#     total_final_results["negative_precision"] = negative_precision
#     total_final_results["negative_recall"] = negative_recall
#     total_final_results["negative_f1"] = negative_f1
    
#     for metric in ["precision","recall","f1","negative_precision","negative_recall","negative_f1"]:
#         hallucination_type_final_results[metric] = {}
#     for classification in classification_types:
#         TP = halucination_specified_dict["f1_matrix"][classification]["TP"]
#         FP = halucination_specified_dict["f1_matrix"][classification]["FP"]
#         FN = halucination_specified_dict["f1_matrix"][classification]["FN"]
#         TN = halucination_specified_dict["f1_matrix"][classification]["TN"]
#         precision = TP / (TP + FP) if (TP + FP) != 0 else -1
#         recall = TP / (TP + FN) if (TP + FN) != 0 else -1
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else -1
#         negative_precision = TN / (TN + FN) if (TN + FN) != 0 else -1
#         negative_recall = TN / (TN + FP) if (TN + FP) != 0 else -1
#         negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if (negative_precision + negative_recall) != 0 else -1
#         hallucination_type_final_results["precision"][classification] = precision
#         hallucination_type_final_results["recall"][classification] = recall
#         hallucination_type_final_results["f1"][classification] = f1
#         hallucination_type_final_results["negative_precision"][classification] = negative_precision
#         hallucination_type_final_results["negative_recall"][classification] = negative_recall
#         hallucination_type_final_results["negative_f1"][classification] = negative_f1
    
#     res = dict(
#         total_hallucination_results=total_final_results,
#         hallucination_type_results=hallucination_type_final_results,
#         validitiy_rate=validitiy_rate,
#         detailed_logs=detailed_logs,
#     )
#     return res




# def eval_on_hallucination_step(hallucination_steps, labels, redundency_label=False):
#     ## Important: hallucination_steps are 0-indexed
#     hallucination_steps = [i-1 for i in hallucination_steps]
#     ## Important: hallucination_steps are 0-indexed
#     if redundency_label:
#         POSITIVE_LABEL = 0
#         NEGATIVE_LABEL = 1
#     else:
#         POSITIVE_LABEL = 1
#         NEGATIVE_LABEL = 0

#     correct_step_acc = []
#     wrong_step_acc = []
#     total_step_acc = []
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
    
#     first_error_location = min(hallucination_steps) if len(hallucination_steps)>0 else -1
#     first_error_acc = None
#     for idx in range(len(labels)):
        
#         if idx == first_error_location:
#             if labels[idx] == NEGATIVE_LABEL:
#                 first_error_acc = 1
#             else:
#                 first_error_acc = 0
        
#         if idx in hallucination_steps:
#             if labels[idx] == POSITIVE_LABEL:
#                 wrong_step_acc.append(0)
#                 total_step_acc.append(0)
#                 FP += 1
#             else:
#                 wrong_step_acc.append(1)
#                 total_step_acc.append(1)
#                 TN += 1
#         else:
#             if labels[idx] == POSITIVE_LABEL:
#                 correct_step_acc.append(1)
#                 total_step_acc.append(1)
#                 TP += 1
#             else:
#                 correct_step_acc.append(0)
#                 total_step_acc.append(0)
#                 FN += 1
                
    
#     correct_step_acc_value = sum(correct_step_acc)/len(correct_step_acc) if len(correct_step_acc)>0 else -1
#     wrong_step_acc_value = sum(wrong_step_acc)/len(wrong_step_acc) if len(wrong_step_acc)>0 else -1
#     total_step_acc_value = sum(total_step_acc)/len(total_step_acc) if len(total_step_acc)>0 else -1
#     model_response_acc = sum(labels)/len(labels) if len(labels)>0 else -1
    
#     return dict(
#         correct_step_acc=correct_step_acc_value,
#         wrong_step_acc=wrong_step_acc_value,
#         total_step_acc=total_step_acc_value,
#         first_error_acc=first_error_acc,
#         model_response_acc=model_response_acc,
#         f1_matrix = dict(TP=TP,FP=FP,TN=TN,FN=FN),
        
        
#         correct_step_acc_list=correct_step_acc,
#         wrong_step_acc_list=wrong_step_acc,
#         total_step_acc_list=total_step_acc,
#         first_error_acc_list=[first_error_acc] if first_error_acc is not None else [],
#         model_response_acc_list=[model_response_acc] if model_response_acc != -1 else [],
#     )