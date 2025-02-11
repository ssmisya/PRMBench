from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerator_logger
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils.log_utils import get_logger
from ..utils.model_utils import remove_step_prefix
from ..utils.utils import *
from .abstract_model import prm

# accelerate_logger = logging.getLogger("debug")
# accelerate_logger.setLevel(logging.DEBUG)
logger = get_logger(__name__)
class PUREPRM(prm):
    def __init__(
            self,
            pretrained = "/mnt/petrelfs/chengjie/ceph2/qwen25-math-7b-PRM800k-bs128-lr1e-6-epoch-1-stage2",
            redundancy_threshold = 0.15,
            validity_threshold = 0.0,
        ) -> None:
        super().__init__(
            validity_threshold=validity_threshold, 
            redundancy_threshold=redundancy_threshold,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, 
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            pretrained, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
                
        self.accelerator = Accelerator()

        self.step_separator = "\n\n"
        self.step_separator_token_id = self.tokenizer.encode(self.step_separator)[0]

    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        ## build model-specialized input
        input_ids = self.tokenizer(
            question, add_special_tokens=False, return_tensors='pt')['input_ids']
        score_ids = []
        for step in steps:
            step = remove_step_prefix(step)
            step_ids = self.tokenizer(
                step, add_special_tokens=False, return_tensors='pt')['input_ids']
            input_ids = torch.cat(
                [input_ids, step_ids, self.step_separator_token_id], dim=-1)
            score_ids.append(input_ids.size(-1) - 1)
        
        input_ids = input_ids.squeenze()
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        token_mask[score_ids] = True
        import ipdb; ipdb.set_trace()
        
        res = dict(
            idx = data_idx,
            input_ids = input_ids,
            token_mask = token_mask,
        )
        return res
    
    def respond(self, dataloader) -> List[Tuple[float, bool]]:
        self.model, dataloader = self.accelerator.prepare(self.model, dataloader)
        self.accelerator.wait_for_everyone()
        self.model.eval()
        gen_kwargs = dataloader.dataset.gen_kwargs
        progress_bar = tqdm_rank0(len(dataloader), desc="Model Responding")
        if len(dataloader) == 0:
            self.accelerator.wait_for_everyone()
            return
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                
                idx = batch['idx']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_mask = batch['token_mask']
                # print(f"data device: {input_ids.device}, current device: {self.accelerator.device}")
                scores = self.model(input_ids,
                                    attention_mask,).logits
                step_reward = make_step_rewards(scores, token_mask)
                
                for i in range(len(idx)):
                    idx_item = idx[i]
                    try:
                        step_level_validity_scores = step_reward[i]
                        score_dict = dict(
                            step_level_validity_scores=step_level_validity_scores,
                            step_level_validity_labels=[item > self.validity_threshold for item in step_level_validity_scores],
                        )
                        res = dict(scores=score_dict, idx=idx_item)
                    except:
                        logger.error(f"Error in processing idx: {idx[i]}")
                        res = dict(scores=dict(), idx=idx_item,validity=False)
                        
                    dataloader.dataset.store_results(res)
                if progress_bar is not None:
                    progress_bar.update(1)
        
        self.accelerator.wait_for_everyone()
        
        
def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        usefule_probs = sample[sample != 0].view(-1, 2) # valid_tokens, num_labels
        step_rewards = usefule_probs[:, 1] - usefule_probs[:, 0]
        non_zero_elements_list = step_rewards.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res