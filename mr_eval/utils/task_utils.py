from .utils import *
from box import Box
        
def load_task_config(file_path):
    task_config = load_yaml_file(file_path)
    task_config = Box(task_config)
    return task_config

        
        
def load_data_function_default(task_config):
    return load_jsonl_data_function_default(task_config)

def load_jsonl_data_function_default(task_config):
    task_name = task_config["task_name"]
    dataset_type = task_config["dataset_type"]
    if dataset_type == "jsonl":
        dataset_path = task_config["dataset_path"]
        meta_data = process_jsonl(dataset_path)
    elif dataset_type == "json":
        dataset_path = task_config["dataset_path"]
        meta_data = load_json_file(dataset_path)
    else:
        raise ValueError(f"dataset_type {dataset_type} not supported")
    return meta_data

def load_dir_of_jsonl_data_function_default(task_config):
    task_name = task_config["task_name"]
    dataset_type = task_config["dataset_type"]
    dataset_path = task_config["dataset_path"]
    assert dataset_type == "dir_of_jsonl"
    assert os.path.isdir(dataset_path)
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jsonl")]
    meta_data = []
    for file in files:
        meta_data.extend(process_jsonl(file))
    return meta_data