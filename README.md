

<p align="center" width="80%">
<img src="./docs/assets/main_logo.png"  width="70%" height="70%">
</p>

# PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models



🏠 [PRMBench Homepage](https://prmbench.github.io/) | 🤗 [Huggingface Datasets](https://huggingface.co/datasets/hitsmy/PRMBench_Preview) | 📑 [Paper](https://arxiv.org/abs/2501.03124) | 📚 [Documentation](docs/document.md)



<p align="center" width="80%">
  <img src="./docs/assets/main_fig.svg" width="100%" height="100%">
</p>
<p align="center" style="font-size: 14px; color: gray;">
  <em>An overview of our <b>PRMBench</b>. The left part illustrates our data curation procedure. In the right part of the figure, we showcase demonstrations of our evaluation subjects and the relative performance of tested models.</em>
</p>
<!-- <p align="center" width="80%">
<img src="./docs/assets/main_fig.svg"  width="70%" height="70%">
</p>
<p>An overview of our $PRMBench$. The left part illustrates our data curation procedure. In the right part of the figure, we showcase demonstrations of our evaluation subjects and the relative performance of tested models. </p> -->

## News
[2025-01-16] We added Qwen2.5-Math-PRM-7B implementations to PRM Eval ToolKit, which becomes the new SOTA for open-source PRMs in PRMBench.

[2025-01-08] We released our paper, code, dataset, and project page.

## PRM Eval ToolKit
> Accelerating the development of process-level reward models (PRMs) with `mr_eval`

PRM Eval ToolKit comprises an automated evaluation framework `mr_eval`, along with a data generation and annotation framework `mr_annotate`. This is also the official github repo for $PRMBench$. The visualization scripts for $PRMBench$ can be found in `mr_visualize`.

`mr_eval` is an auto PRM evaluation model which is adaptable to custom datasets and models, ensuring broad accessibility.



## Installation

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

## Example Usages

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

## PRMBench LeaderBoard

| Model | Overall| NR. | NCL. | Avg (simplicity) | ES. | SC. | DC. | CI. | Avg (soundness) | PS. | DR. | MS. | Avg (sensitivity)  |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|------- |
| [Skywork-PRM-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B) | 31.7 | 31.4 | 35.8 | 33.6 | 32.4 | 25.7 | 26.0 | 30.2 | 28.6 | 33.1 | 32.3 | 81.1 | 48.8 
| [Skywork-PRM-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B) | 36.2 | 35.7 | 41.2 | 38.4 | 36.7 | 29.1 | 30.6 | 34.4 | 32.7 | 36.8 | 37.4 | 88.8 | 54.3 
| [Llemma-PRM800k-7B](https://huggingface.co/ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf) | 52.0 | 49.3 | 53.4 | 51.4 | 56.4 | 47.1 | 46.7 | 53.3 | 50.9 | 51.0 | 53.5 | 93.6 | 66.0 
| [Llemma-MetaMath-7B](https://huggingface.co/ScalableMath/llemma-7b-prm-metamath-level-1to3-hf) | 50.5 | 50.2 | 50.5 | 50.3 | 51.9 | 47.6 | 44.4 | 52.1 | 49.0 | 50.5 | 51.3 | 96.0 | 66.0 
| [Llemma-oprm-7B](https://huggingface.co/ScalableMath/llemma-7b-oprm-prm800k-level-1to3-hf) | 50.3 | 48.7 | 49.3 | 49.0 | 54.2 | 46.8 | 44.5 | 53.5 | 49.8 | 49.2 | 51.3 | 91.8 | 64.1 
| [MATHMinos-Mistral-7B](https://github.com/KbsdJames/MATH-Minos) | 54.2 | 48.8 | _54.0_ | 51.4 | 57.0 | 52.1 | 50.7 | 57.8 | 54.4 | 52.8 | 55.8 | 91.1 | 66.5 
| [MathShepherd-Mistral-7B](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm) | 47.0 | 44.0 | 50.3 | 47.1 | 49.4 | 44.5 | 41.3 | 47.7 | 45.7 | 47.2 | 48.6 | 86.1 | 60.7 
| [ReasonEval-7B](https://huggingface.co/GAIR/ReasonEval-7B) | 60.0 | **61.0** | 50.1 | **55.6** | 62.1 | _65.9_ | _61.5_ | 65.9 | _63.8_ | 55.6 | 57.9 | 99.5 | 71.0 
| [RLHFlow-PRM-Mistral-8B](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data) | 54.4 | 46.1 | 47.3 | 46.7 | 56.6 | 55.1 | 54.4 | 63.8 | 57.5 | 51.5 | 56.2 | 97.9 | 68.5 
| [RLHFlow-PRM-Deepseek-8B](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data) | 54.2 | 46.4 | 48.9 | 47.6 | 55.7 | 55.0 | 53.2 | 66.2 | 57.5 | 49.0 | 55.4 | **99.8** | 68.1 
| [ReasonEval-34B](https://huggingface.co/GAIR/ReasonEval-34B) | _60.5_ | _54.8_ | 48.1 | 51.5 | _66.4_ | 60.3 | 57.8 | _67.5_ | 63.0 | **57.7** | _64.3_ | 97.2 | _73.1_ 
| [Qwen2.5-Math-PRM-7B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) | **65.5** | 49.1 | **55.2** | _52.1_ | **71.8** | **67.2** | **66.3** | **78.5** | **71.0** | _57.7_ | **69.2** | _99.7_ | **75.6** |
| **Avg.** | **51.4** | **47.1** | **48.7** | **47.9** | **54.2** | **49.7** | **48.1** | **55.9** | **52.0** | **49.3** | **52.8** | **93.6** | **65.2** |
| [GPT-4o](https://openai.com/index/hello-gpt-4o/) | 66.8 | 57.0 | 62.4 | 59.7 | 72.0 | _69.7_ | 70.7 | 71.1 | 70.9 | **62.5** | 65.7 | 99.2 | **75.8** 
| [o1-mini](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/)\$^\dagger$ | _68.8_ | 65.6 | _63.7_ | _64.6_ | **74.5** | 67.7 | **73.8** | **72.3** | **72.1** | _61.8_ | 64.8 | **100.0** | _75.5_ 
| [Gemini-2.0-flash-exp](https://deepmind.google/technologies/gemini/flash/) | 66.0 | _67.2_ | 58.1 | 62.7 | 70.4 | 65.7 | 66.0 | 67.3 | 67.3 | 61.8 | **66.2** | 98.2 | 75.4 
| [Gemini-2.0-thinking-exp-1219](https://ai.google.dev/gemini-api/docs/thinking-mode) | **68.8** | **68.5** | **63.8** | **66.2** | _72.9_ | **71.3** | _71.0_ | _71.8_ | _71.8_ | 60.3 | _65.7_ | _99.8_ | 75.3 
| **Avg.** | **67.6** | **64.6** | **62.0** | **63.3** | **72.4** | **68.6** | **70.4** | **70.7** | **70.5** | **61.6** | **65.6** | **99.3** | **75.5** |




The best performance for each category and task is in **bold**, while the second-best performance is shown in _italic_. $^\dagger$: To reduce costs, we evaluated only a subset of 394 samples for o1 series models.

## Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md).

## Data Format for PRMBench

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
