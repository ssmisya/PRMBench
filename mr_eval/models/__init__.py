import importlib
import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "reasoneval": "ReasonEval",
    "math_shepherd": "MathShepherd",
    "llemma7b_prm": "Llemma7bPRM",
    "mathminos_mistral": "MathMinos_Mistral",
    "openai_models": "OpenaiModels",
    "llama3_1_8b_prm": "LLaMA318BPRM",
    "skywork_prm":"SkyworkPRM",
    "gemini_models": "GeminiModels",
    "qwen_qwq": "QwenQwQ",
    "qwen_prm": "QwenPRM",
    "vllm_models": "VllmModels",
    "pure_prm": "PUREPRM",
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    try:
        module = __import__(f"mr_eval.models.{model_name}", fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise
