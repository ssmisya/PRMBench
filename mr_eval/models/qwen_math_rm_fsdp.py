import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    ShardingStrategy,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm




class InferenceManager:
    def __init__(self, model_name, checkpoint_path=None):
        self.setup_dist()
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.local_rank = int(os.environ["LOCAL_RANK"])
        
        # 初始化模型和分词器
        self.init_model_and_tokenizer()
    
    def setup_dist(self):
        dist.init_process_group("nccl")
        torch.cuda.set_device(dist.get_rank())
    
    def init_model_and_tokenizer(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # FSDP包装
        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ),
        )
        
        # 加载检查点
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path)
            self.model.load_state_dict(state_dict)
    
    def inference(self, texts, batch_size=4, max_length=100):
        """批量推理"""
        self.model.eval()
        results = []
        
        # 创建进度条（只在主进程）
        if self.local_rank == 0:
            pbar = tqdm(range(0, len(texts), batch_size), desc="Inferencing")
        else:
            pbar = range(0, len(texts), batch_size)
        
        for i in pbar:
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                if self.local_rank == 0:
                    decoded_outputs = [
                        self.tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    results.extend(decoded_outputs)
        
        return results if self.local_rank == 0 else None
    
    def cleanup(self):
        dist.destroy_process_group()

def main():
    # 配置
    model_name = "your_model_name"
    checkpoint_path = "path/to/checkpoint.pt"
    
    # 创建推理管理器
    inference_manager = InferenceManager(model_name, checkpoint_path)
    
    # 准备输入数据
    texts = [
        "First input text",
        "Second input text",
        # ... more texts
    ]
    
    # 执行推理
    results = inference_manager.inference(texts, batch_size=4)
    
    # 打印结果（只在主进程）
    if dist.get_rank() == 0:
        for i, result in enumerate(results):
            print(f"Input {i}:")
            print(f"Generated: {result}\n")
    
    # 清理
    inference_manager.cleanup()

if __name__ == "__main__":
    main()