


# Documentations For PRM Eval ToolKit 

## 1. Config File Formatting

**We released some example scripts/configs to demonstrate how to use our toolkit. You can find them in the `mr_eval/scripts` directory.**


You can organize your config as a list of dict or a single dict. It's recommend to use a yaml file. 
The 
```yaml
- model_args:
    # The model series you want to test, must be the same with the file name under mr_eval/models
    model: reasoneval 
    # The arguments that you want to pass to the models, split by a comma.
    model_args: pretrained=GAIR/ReasonEval-7B,model_size=7B,redundancy_threshold=0.15
    # The batch size if you want to use batch inference.
    batch_size: 2
  task_args:
    # The task names you want to evaluate, split by a comma.
    task_name: prmtest_classified
    # checkpoint settings, organize them as a dict
    # taskname: ckpt_file path
    resume_from_ckpt:
      prmtest_classified: ./logs/generated/ckpt/reasoneval7b_prm800k_classified.jsonl
    save_to_ckpt:
      prmtest_classified: ./logs/generated/ckpt/reasoneval7b_prm800k_classified.jsonl
  script_args:
    verbosity: INFO
    # final result output path
    output_path: ./logs/prmtest_classified/reasoneval7b.jsonl
```
After setting down the config, please run $PRMEval$ as:

```bash
accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval \
--config ${config_file}
```

Our batch inference and multi-gpu parallel is inferenced based on huggingface accelerate, so please prepare a accelerate config and run based on it.
An example accelerate config is:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

But notice that when testing api modes (e.g. gemini and openai series models), the batch size must be set at 1 and do not use multi-process parallel.


## 2.Introduction to our basic framework

Our `PRMEval` framework is consisted with two important concepts: `task` and `model`. You can add custom tasks or models to customize your own evaluation framework.
The tasks and models are connected throuth a pytorch dataset, whose basic implementation can be found at `mr_eval/tasks/base_dataset`. The **data loading logic**(`load_data_function()`) and **evaluation logic**(`evaluate_function()`) is implemeted by `task` and the **get data instance logic**(`getitem_function(self,meta_data,index)`) is implemented by `model`.

The results of the evaluation will be staged in `base_dataset`, so you can call `dataloader.dataset.store_results(res)` to stage the results temporarily. After the whole evaluation process, the evaluation process will call `evaluate_function()` to get the final results.

## 3.How to add a new model?

1. Implement your model inference script under `mr_eval/models`

2. Wrap your model inference code with base class `prm`

3. Implement the `getitem_function` and `respond` function in your model class

4. register your model in `AVAILABLE_MODELS` in `mr_eval/models/__init__.py`, the key should be the same with the `model` in your config file and your implement python file name. The value should be the class name of your model.

Notes:

1. When implementing the `getitem_function`, you should return a dict whose keys and values are decided by yourself. You can design the data structure of the dict based on your model `respond` logic.

2. Generally speaking, the temprary results is formatted as:
```python
score_dict = dict(step_level_redundancy_scores=step_level_redundancy_scores, 
                  step_level_validity_scores=step_level_validity_scores,
                  step_level_redundancy_labels=[item > self.redundancy_threshold for item in step_level_redundancy_scores], 
                  step_level_validity_labels=[item > self.validity_threshold for item in step_level_validity_scores],
                  solution_level_redundancy_scores= max(step_level_redundancy_scores), 
                  solution_level_validity_scores=min(step_level_validity_scores),
                  solution_level_redundancy_labels=max(step_level_redundancy_scores)>self.redundancy_threshold,
                  solution_level_validity_labels=min(step_level_validity_scores)>self.validity_threshold
)
res = dict(scores=score_dict, idx=idx_item)
```
It is dealed between model and task, which means you can customize them. For example, PRMBench only uses `step_level_validity_scores`, `step_level_redundancy_scores` `step_level_redundancy_labels` and `step_level_validity_labels` to evaluate the model.

3. When finish one round of batch inference, you should call `dataloader.dataset.store_results(res)` to store the results in the dataset sequentially.

4. An example template is:

```python
from .abstract_model import prm
class GeminiModels(prm):
    def __init__(
            self,
            model_name = "your-model-name",
        ) -> None:
        super().__init__(validity_threshold=validity_threshold, redundancy_threshold=redundancy_threshold)
        # your initialize scripts

    def getitem_function(self,meta_data,index) -> Dict:
        pass
    
    def respond(self, dataloader) -> None:
        pass

    
```
## 4.How to add a new task?

1. Implement your task under `mr_eval/tasks/your-task-name`
2. Implement your `mr_eval/tasks/your-task-name/config.yaml` and `mr_eval/tasks/your-task-name/task.py`
3. In `task.py`, implement the `load_data_function` and `evaluate_function`
4. No need to register your task, but make sure the `task_name` in your config file is the same with the folder name of your task, that is `your-task-name` in this demo.
