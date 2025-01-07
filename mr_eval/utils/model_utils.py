import os,sys,torch
from mr_eval.utils import *
sys.path.append("/mnt/petrelfs/songmingyang/code/reasoning/MR_Hallucination/ref/ReasonEval/codes")
from examples import *

import json, re
import torch.nn as nn
from transformers import MistralModel, MistralPreTrainedModel, LlamaModel, LlamaPreTrainedModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from copy import deepcopy
from tqdm import tqdm


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
    steps = prm_item["label"]["steps"]
    best_answers = []
    for step in steps:
        if step["human_completion"] is not None and step["chosen_completion"] is None:
            best_answers.append(step["human_completion"])
        elif step["chosen_completion"] is not None:
            best_answers.append(step["completions"][step["chosen_completion"]])
        else:
            print(f"skipped one step")
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



def initialize_reasoneval_model(model_name_or_path="/mnt/petrelfs/songmingyang/songmingyang/model/reasoning/ReasonEval-7B",model_size="7B"):
    # Loading the model
    model_file_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    if model_size == '7B':
        model = ReasonEval_7B.from_pretrained(model_file_path)
    elif model_size == '34B':
        model = ReasonEval_34B.from_pretrained(model_file_path)
    else:
        raise ValueError(f"Invalid model size: {model_size}")
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model,tokenizer

def reasoneval_inference(question,reasoning_steps,model,tokenizer):
    # Preprocessing input
    PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
    step_separator = f"{tokenizer.pad_token}"
    combined_steps = ""
    for steps in reasoning_steps:
        combined_steps += steps + step_separator
    prompt = PROMPT_FORMAT.format(input=question)
    tokenized_result = tokenizer(prompt + step_separator + combined_steps)['input_ids']

    ## Separating labels and adjusting token IDs
    separator_token_id = tokenizer(step_separator)['input_ids'][-1]
    labeled_token_indices = []
    adjusted_token_ids = []
    separator_count = 0
    for idx, token_id in enumerate(tokenized_result):
        if token_id == separator_token_id:
            labeled_token_indices.append(idx - 1 - separator_count)
            separator_count += 1
        else:
            adjusted_token_ids.append(token_id)
    if isinstance(model,ReasonEval_7B):
        adjusted_token_ids = [1] + adjusted_token_ids # Adjusting to recover the first token_ids of the sentences
        adjusted_token_ids=torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[2:]  # Adjusting to skip the first two separator (begining and endding of the problems)
    elif isinstance(model,ReasonEval_34B):
        adjusted_token_ids=torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[1:]  # Adjusting to skip the first separator (endding of the problems)
    else:
        raise ValueError(f"Invalid model size!")
    assert len(labeled_token_indices) == len(reasoning_steps)

    attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
    # Evaluating reasoning steps using ReasonEval
    with torch.no_grad():
        adjusted_token_ids = adjusted_token_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        reasoning_scores = model(adjusted_token_ids, attention_mask)[0,labeled_token_indices , :]
        scores = torch.softmax(reasoning_scores, dim=-1).tolist()

    # Calculating the validity and redundancy scores
    ## score: [p_{negative}, p_{neutral}, p_{positive}]

    ## S_{validity} = p_{neutral} + p_{positive}
    step_level_validity_scores =  [(score[1] + score[2]) for score in scores]
    print(f"step_level_validity_scores: {step_level_validity_scores}")
    # ReasonEval-7B for the example: [0.9492, 0.7863 ,0.2520, 0.7860, 0.9125, 0.5916, 0.4494, 0.8189, 0.8240, 0.2671]
    # ReaspnEval-34B for the example: [0.9360, 0.6813 ,0.0720, 0.2811, 0.4531, 0.1122, 0.1328, 0.2026, 0.2265 0.1163]

    ## S_{redundancy} = p_{neutral}
    step_level_redundancy_scores = [score[1] for score in scores]
    print(f"step_level_redundancy_scores: {step_level_redundancy_scores}")
    # ReasonEval-7B for the example: [0.4433, 0.1287, 0.0397, 0.0789, 0.0789, 0.0509, 0.0487, 0.0702, 0.0955, 0.0120]
    # ReasonEval-34B for the example: [0.6060, 0.1682, 0.0258, 0.1044, 0.1604, 0.0404, 0.0447, 0.0507, 0.0492, 0.0236]
    
    solution_level_validity_scores = min(step_level_validity_scores)
    print(f"solution_level_validity_scores: {solution_level_validity_scores}")
    solution_level_redundancy_scores = max(step_level_redundancy_scores)
    print(f"solution_level_validity_scores: {solution_level_redundancy_scores}")
    return step_level_validity_scores,step_level_redundancy_scores,solution_level_validity_scores,solution_level_redundancy_scores

def score_list_to_str(score_list):
    valid2_list = [str(round(i,2)) for i in score_list]
    res =  ", ".join(valid2_list)
    return res


def clean_str(input_str):
    res_str = deepcopy(input_str)
    res_str = re.sub(r'\\+([^\\\s])', r'\\\\\1', res_str)
    res_str = re.sub(r'\\+([\s])', r'\\\\\\\\\1', res_str)
    return res_str

def remove_comments_from_json(json_string):
    """
    移除 JSON 字符串中的单行和多行注释。
    """

    # 匹配 // 和 # 开头的注释，并移除
    return re.sub(r'//.*?$|#.*?$', '', json_string, flags=re.MULTILINE)

def extract_nested_json(text):
    """
    提取嵌套大括号内的 JSON 数据，移除注释后解析。
    Args:
        text (str): 包含 JSON 的文本。
    Returns:
        dict or list or None: 解析成功返回 JSON 数据，失败返回 None。
    """
    stack = []  # 用来记录大括号的匹配
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if not stack:  # 当栈为空时，记录第一个大括号的位置
                start = i
            stack.append("{")  # 压栈
        elif char == "}":
            stack.pop()  # 出栈
            if not stack:  # 当栈为空时，表示找到完整 JSON
                try:
                    # 提取完整 JSON 字符串
                    json_str = text[start:i+1]
                    # 移除注释
                    json_cleaned = remove_comments_from_json(json_str)
                    # 尝试解析为 JSON 对象
                    return json.loads(json_cleaned)
                except json.JSONDecodeError as e:
                    continue  # 如果解析失败，跳过并继续查找
    return None  # 如果未找到完整 JSON，则返回 None

def process_policy_lm_evaluation_response(response):
    """ process the response STRING from the language model"""
    try:
        json_object = extract_nested_json(response)
        assert json_object is not None
        assert "validity" in json_object and "redundancy" in json_object
        return json_object
    except :
        print(f"Invalid JSON Str, response: {response}")
        return None


def remove_step_prefix(text):
    """
    去掉以 'Step x. ' 或 'step x. ' 或 'x. ' 开头的部分，其中 x 是数字
    """
    text = text.strip()
    return re.sub(r"^(Step\s*\d+\.\s*|\d+\.\s*)", "", text, flags=re.IGNORECASE)

def find_subsequence(tensor, subsequence):
    """
    在张量中定位子串的位置。

    Args:
        tensor (torch.Tensor): 主张量。
        subsequence (torch.Tensor): 子串张量。

    Returns:
        List[int]: 子串在主张量中的起始位置索引列表。
    """
    main_len = tensor.size(0)  # 主张量的长度 (假设是二维张量，取列数)
    sub_len = subsequence.size(0)  # 子串的长度

    positions = []  # 存储匹配的起始位置
    for i in range(main_len - sub_len + 1):  # 滑动窗口遍历
        # 比较切片是否与子串相等
        if torch.equal(tensor[i:i+sub_len], subsequence):
            positions.append(i)
    return positions

