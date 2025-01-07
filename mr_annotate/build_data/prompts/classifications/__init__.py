import importlib

classifications = ["circular", "confidence", "counterfactual", "step_contradiction", "domain_inconsistency", "redundency", "missing_condition", "deception"]
few_shots_names = ["fewshot_q1", "fewshot_a1", "fewshot_q2", "fewshot_a2"]
__all__ = []

for classification in classifications:
    for few_shot_name in few_shots_names:
        object_name = f"{classification}_{few_shot_name}"
        module = importlib.import_module(f".{classification}", package=__name__)  # 动态导入模块
        fs_obj = getattr(module, object_name) 
        globals()[object_name] = fs_obj
        __all__.append(object_name)


fewshot_dicts = {classification: [(globals()[f"{classification}_fewshot_q1"], globals()[f"{classification}_fewshot_a1"]), 
                                  (globals()[f"{classification}_fewshot_q2"], globals()[f"{classification}_fewshot_a2"])] for classification in classifications}
__all__.append("fewshot_dicts")