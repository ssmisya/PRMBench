from typing import List, Tuple

import torch
from accelerate import Accelerator
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..utils.log_utils import get_logger
from ..utils.utils import *
from .abstract_model import prm

logger = get_logger(__name__)


class PUREPRM(prm):
    def __init__(
            self,
            pretrained = "jinachris/Qwen2.5-Math-7B-PRM800K",
            redundancy_threshold = 0.0,  # not used?
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
        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
                
        self.accelerator = Accelerator()

        self.step_separator = "\n\n"
        self.step_separator_token_id = self.tokenizer(
            self.step_separator, add_special_tokens=False, return_tensors='pt')['input_ids']

    def getitem_function(self,meta_data,index):
        data_idx = meta_data[index]["idx"]
        steps = meta_data[index]["steps"]
        question = meta_data[index]["question"]
        
        ## build model-specialized input
        input_ids = self.tokenizer(
            question, add_special_tokens=False, return_tensors='pt')['input_ids']
        score_ids = []
        for step in steps:
            step_ids = self.tokenizer(
                step, add_special_tokens=False, return_tensors='pt')['input_ids']
            input_ids = torch.cat(
                [input_ids, step_ids, self.step_separator_token_id], dim=-1)
            score_ids.append(input_ids.size(-1) - 1)
        
        input_ids = input_ids.squeeze()
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        token_mask[score_ids] = True
        
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
        progress_bar = tqdm_rank0(len(dataloader), desc="Model Responding")
        if len(dataloader) == 0:
            self.accelerator.wait_for_everyone()
            return
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                idx = batch['idx']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                # right pad token mask
                token_mask_ = batch['token_mask']
                token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                bs = input_ids.size(0)
                for i in range(bs):
                    token_mask[i, attention_mask[i].to(bool)] = token_mask_[i][:attention_mask.size(1)]
                assert torch.all(input_ids[token_mask] == self.step_separator_token_id.item())

                scores = self.model(input_ids, attention_mask).logits
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
    all_scores_res = []
    for sample, token_mask in zip(logits, token_masks):
        probs = sample[token_mask].softmax(dim=-1)
        process_reward = probs[:, 1] - probs[:, 0]
        all_scores_res.append(process_reward.cpu().tolist())
    return all_scores_res