
<!--
<p align="center" width="80%">
<img src="./docs/assets/main_logo.png"  width="70%" height="70%">
</p>
-->
# PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models



<!-- 🏠 [PRMBench Homepage](https://prmbench.github.io/) | 🤗 [Huggingface Datasets](https://huggingface.co/datasets/hitsmy/PRMBench_Preview) | 📑 [Paper](https://arxiv.org/abs/2501.03124) | 📚 [Documentation](docs/document.md) -->
<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="https://prmbench.github.io/" style="text-decoration: none; font-weight: bold;">🌻 Homepage</a> •
    <a href="https://huggingface.co/datasets/hitsmy/PRMBench_Preview" style="text-decoration: none; font-weight: bold;">🤗 Data</a> •
    <a href="https://arxiv.org/abs/2501.03124" style="text-decoration: none; font-weight: bold;">📑 Paper</a> •
    <a href="docs/document.md" style="text-decoration: none; font-weight: bold;">📖 Documentation</a>
  </p>
</div>

<p align="center" width="80%">
  <img src="./docs/assets/main_fig.svg" width="70%" height="70%">
</p>
<p align="center" style="font-size: 14px; color: gray;">
  <em>An overview of our <b>PRMBench</b>. The left part illustrates our data curation procedure. In the right part of the figure, we showcase demonstrations of our evaluation subjects and the relative performance of tested models.</em>
</p>


## News
🚀 [2025-02-07] We've added a collection of results from various open-source LLMs used as critic models. You can check out the results [here](https://prmbench.github.io/#leaderboard_test).

⚡ [2025-02-04] PRM Eval ToolKit now supports inferencing with VLLM.

🛠️ [2025-01-20] PRM Eval ToolKit now supports FSDP and DeepSpeed, which means you can evaluate 72B models utilizing our framework. Moreover, we add results of Qwen2.5-Math-PRM-72B on PRMBench.

🛠️ [2025-01-16] We added Qwen2.5-Math-PRM-7B implementations to PRM Eval ToolKit, which becomes the new SOTA for open-source PRMs in PRMBench.

✨ [2025-01-08] We released our paper, code, dataset, and project page.

## PRM Eval ToolKit
> Accelerating the development of process-level reward models (PRMs) with `mr_eval`

PRM Eval ToolKit comprises an automated evaluation framework `mr_eval`, along with a data generation and annotation framework `mr_annotate`. This is also the official github repo for $PRMBench$. The visualization scripts for $PRMBench$ can be found in `mr_visualize`.

`mr_eval` is an auto PRM evaluation framework adaptable to custom datasets and models, ensuring broad accessibility.



## 🛠️ Installation

```bash
git clone https://github.com/ssmisya/PRMBench
cd PRMBench

# Optional: create a conda virtual environment
conda create -n mr_eval python=3.10
conda activate mr_eval

# Install dependencies, you can install pytorch according to your CUDA version
pip install -r requirements.txt
pip install -e .
```

## 📝 Example Usages

We released some example scripts/configs to demonstrate how to use our toolkit. You can find them in the `mr_eval/scripts` directory.

**Evaluation of ReaonEval-7B on PRMBench Directly**

A simple way to run Mr Eval

```bash
accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval \
--model reasoneval \
--model_args pretrained=GAIR/ReasonEval-7B \
--task_name prmtest_classified \
--verbosity INFO \
--output_path ./scripts/logs/prmtest_classified/reasoneval_7b.jsonl
```

Note that task `prmtest_classified` is our default task for PRMBench.

**Evaluation of ReaonEval-7B on PRMBench Using a Config File**

We strongly recommend that using a config file to evaluate PRMs.

```bash
accelerate launch  --config_file  ${accelerate_config} \
-m mr_eval \
--config ${config_file}
```

Config file example:

```yaml
- model_args:
    model: reasoneval
    model_args: pretrained=GAIR/ReasonEval-7B
    batch_size: 2
  task_args:
    task_name: prmtest_classified
    resume_from_ckpt:
      prmtest_classified: ./logs/generated/ckpt/reasoneval7b_prm800k_classified.jsonl
    save_to_ckpt:
      prmtest_classified: ./logs/generated/ckpt/reasoneval7b_prm800k_classified.jsonl
  script_args:
    verbosity: INFO
    output_path: ./logs/prmtest_classified/reasoneval7b.jsonl
```

For detailed information and config setting please refer to our [documentation](docs/README.md).

## 🏆 PRMBench LeaderBoard

The leaderboard is available [here](https://prmbench.github.io/#leaderboard_test).

## 📦 Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md).

## 📈 Data Format for PRMBench

In PRMBench, our data format can be formulated as follows:

```python
{
    # Original question
    "original_question": "Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?",
    # Modified question --used for the evaluation
    "modified_question": "Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?",
    # Original process -- the original solution steps
    "original_process": [
        "1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.",
        "2. We have $3p+e=1.24$ and $5p+e=1.82$.",
        "3. To solve this system, let's subtract the first equation from the second equation. This will eliminate e.",
        "4. $5p+e-3p-e=1.82-1.24$.",
        "5. This simplifies to $2p=0.58$. So $p=0.29$.",
        "6. That means a pencil costs 29 cents."
    ],
    # Modified process -- used for the evaluation
    "modified_process": [
        "1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.",
        "2. We have $3p+e=1.24$ and $5p+e=1.82$.",
        "3. Assume a pencil costs 29 cents. Then verify this against the original equations.",
        "4. If $p=0.29$, substitute into one of the equations, for example, $3p+e=1.24$. It gives $3(0.29)+e=1.24$.",
        "5. Solving $0.87+e=1.24$ gives $e=0.37$. Now check $5p+e=1.82$ to confirm.",
        "6. Plug $p=0.29$ and $e=0.37$ into $5p+e=1.82$. We get $5(0.29)+0.37=1.82$.",
        "7. The check confirms that the assumed price of 29 cents per pencil is correct."
    ],
    # Modified steps -- the steps that are modified 
    "modified_steps": [3, 4, 5, 6],
    # Error steps -- the steps that contain errors
    "error_steps": [3, 4, 5],
    # Reason for the error
    "reason": "Steps 3, 4, and 5 introduce circular reasoning by assuming the result ($p = 0.29$) and verifying it rather than deriving it through independent calculations. This creates a fallacious reasoning process since the solution is assumed as a premise and then used to confirm itself.",
    # idx -- unique identifier for the data instance
    "idx": "circular_prm_test_p1_0",
    # question -- the original question
    "question": "Three pencils and a jumbo eraser cost $\\$1.24$. Five pencils and a jumbo eraser cost $\\$1.82$. No prices include tax. In cents, what is the cost of a pencil?",
    # classification -- the classification of the error
    "classification": "circular"
}
```

## Citations

```bibtex
@article{song2025prmbench,
  title={PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models},
  author={Mingyang Song and Zhaochen Su and Xiaoye Qu and Jiawei Zhou and Yu Cheng},
  journal={arXiv preprint arXiv:2501.03124},
  year={2025},
  url={https://arxiv.org/pdf/2501.03124}
}
```
![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-green)
