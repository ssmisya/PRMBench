import json
import yaml
import os
import torch.distributed as dist
from tqdm import tqdm

def load_json_file(filepath):
    '''
        将json文件读取成为列表或词典
    '''
    with open(filepath, 'r',encoding="UTF-8") as file:
        data = json.load(file)
    return data

def write_json_file(data, filepath):
    with open(filepath, 'w',encoding="UTF-8") as f:
        json.dump(data, f, ensure_ascii=False,indent=4)

def process_jsonl(file_path):
    '''
        将jsonl文件转换为装有dict的列表
    '''
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def write_jsonl(data, file_path):
    '''
        将list[dict]写入jsonl文件
    '''
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            line = json.dumps(item,ensure_ascii=False)
            file.write(line + '\n')

def merge_jsonl(input_file_dir, output_filepath):
    '''
        将源文件夹内的所有jsonl文件合并为一个jsonl文件,并保存在output_filepath中
    '''
    filepaths = [os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir)]
    merged_data = []
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            for line in file:
                data = json.loads(line)
                merged_data.append(data)
    
    with open(output_filepath, 'w') as output_file:
        for data in merged_data:
            output_file.write(json.dumps(data) + '\n')

def append_jsonl(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')
        
def load_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [line.strip().replace("\n","") for line in data]
    return data

def write_txt_file(data, filepath):
    for item in data:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(item + '\n')
            
            
def print_rank0(msg):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg)
    else:
        print(msg)

def str2list(input_str):
    if isinstance(input_str,str):
        raw_list = input_str.strip().replace("\n","").split(",")
        new_list = []
        for item in raw_list:
            new_list.append(item.strip())
        return new_list
    elif isinstance(input_str,list):
        return input_str
    else:
        raise TypeError("input_str should be str or list")

def get_two_words(word1,word2):
    if word1 < word2:
        return f"{word1},{word2}"
    else:
        return f"{word2},{word1}"
    
 
def load_yaml_file(filepath):
    with open(filepath, 'r',encoding="UTF-8") as file:
        data = yaml.safe_load(file)
    return data

def write_yaml_file(data, filepath):
    with open(filepath, 'w',encoding="UTF-8") as file:
        yaml.dump(data, file,indent=4)
        
def tqdm_rank0(total, desc):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            pbar = tqdm(total=total, desc=desc)
            return pbar
        else:
            return None
    else:
        pbar = tqdm(total=total, desc=desc)
        return pbar

def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True

def dist_wait_for_everyone():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        
    
    
def gather_dict_lists(local_dict_list):
    '''
        使用all_gather_object收集所有进程的数据
    '''
    if dist.is_available() and dist.is_initialized():
        # 获取总进程数
        world_size = dist.get_world_size()

        # 准备接收对象的空列表，每个进程分配一个 None
        gathered_dict_lists = [None for _ in range(world_size)]

        # 收集所有进程的数据
        dist.all_gather_object(gathered_dict_lists, local_dict_list)

        # 合并所有进程的数据到一个完整的列表
        final_merged_list = [item for sublist in gathered_dict_lists for item in sublist]
        return final_merged_list
    else:
        return local_dict_list

def setup_proxy():
    AD_NAME="songmingyang"
    encrypted_password="dSpydxsxxhKix63HfIFhjwnZLEInXEDawSoMD35G1IT2CygKnHsJqG9ZHbEP"
    new_proxy_address=f"http://{AD_NAME}:{encrypted_password}@10.1.20.50:23128/"
    # 设置环境变量
    os.environ['http_proxy'] = new_proxy_address
    os.environ['https_proxy'] = new_proxy_address
    os.environ['HTTP_PROXY'] = new_proxy_address
    os.environ['HTTPS_PROXY'] = new_proxy_address
    
def setup_openai_proxy():
    new_proxy_address="http://closeai-proxy.pjlab.org.cn:23128"
    # 设置环境变量
    os.environ['http_proxy'] = new_proxy_address
    os.environ['https_proxy'] = new_proxy_address
    os.environ['HTTP_PROXY'] = new_proxy_address
    os.environ['HTTPS_PROXY'] = new_proxy_address
    