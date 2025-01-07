from .classifications import fewshot_dicts
basic_prompt = """
You are a helpful AI assistant that is very good at reasoning and data construction. Now I want to test the ability of process-level reward models to judge whether a step within reasoning process is correct. To do this, please help me build flawed cases by introducing specific types of errors into a given reasoning process.

You will be provided with:

1. A mathematical problem.
2. Its standard correct answer.
3. A correct step-by-step reasoning process used to solve it.

Your task is to modify the question, adjust one or more steps, or introduce additional steps into the original process chain to create a reasoning process that appears plausible but is incorrect. The objective is to simulate flawed solutions by incorporating the specified error detailed after '### Error Type to Introduce'.

### Error Type to Introduce
"""

redundency = """
Redundancy refers to a process that is not the most concise or efficient, as it includes one or more redundant steps that can be removed without affecting the correctness of the overall solution path. For example, if $ A \\to B $ represents a correct inference chain, your task is to introduce one or more redundant steps $ C = {c | c is redundent} $ and reformulate the solution chain as $ A \\to C \\to B $.
"""

circular = """
Circular logic is a specific form of redundancy, characterized by a reasoning chain that starts at a step $ S $, progresses through a sequence of steps, and ultimately loops back to $ S $. Symbolically, this can be expressed as $ S \\to A \\to B \\to S $, where $ S $, $ A $, and $ B $ represent individual reasoning steps. Your task is to modify the reasoning process to introduce such circular logic.
"""

counterfactual="""
A counterfactual step refers to a statement within a reasoning chain that contradicts established ground truth. Such contradictions can arise from relying on outdated theories, omitting critical constraints in a theory, or incorporating erroneous assumptions. Your task is to modify the reasoning process to introduce such counterfactual steps.
"""

step_contradiction="""
Step contradiction refers to a conflict between a specific step and other steps within a reasoning path. Given a reasoning path $ P = {S_1, S_2, \dots, S_n} $, a step contradiction exists if $ S_i \perp S_j $, where $ i, j \in [1, n] $ and $ i \\neq j $. Your task is to modify the reasoning process to introduce such step contradiction steps.
"""

domain_inconsistency="""
Domain inconsistency is a special type of counterfactual. It refers to a step within the reasoning chain that uses a statement or theory valid in other domains or cases but is not valid within the current reasoning chain. Your task is to modify the reasoning process to introduce such domain inconsistency steps.
"""

confidence="""
Confident hallucination is a special type of counterfactual. It refers to a statement within the reasoning chain that contradicts established ground truth and is presented with an overly confident tone. In other words, it involves stating an incorrect statement with unwarranted certainty. Your task is to modify the reasoning process to introduce such confident hallucination steps.
"""

missing_condition="""
Missing condition or prerequisite refers to a flaw in the reasoning chain where critical premises, assumptions, or necessary conditions are absent. This omission results in logical gaps, incomplete reasoning, or biased conclusions. For example, when a missing condition occurs, the model is required to solve the problem through case analysis or further investigation. However, the answer becomes incorrect if the model overlooks the missing condition and proceeds with standard reasoning methods. Your task is to modify the reasoning process to introduce such missing condition error steps.
"""

deception="""
Deception or traps refer to statements that appear to be correct or align with ground truth but are subtly altered to introduce inaccuracies while maintaining the illusion of correctness. Your task is to modify the reasoning process to introduce such deception or trap error steps.
"""


post_prompt = """
### Formatting Instructions

After making the modifications, provide the following structured output:
{
    "original_question": "The original mathematical problem.",
    "modified_question": "The modified problem or original problem
    "original_process": ["original_step 1", "original_step 2", ...],
    "modified_process": ["modified_step 1", "modified_step 2", ...],
    "modified_steps": [1, 5, 7, ...],
    "error_steps": [5, 6, ...],
    "reason": "Explanation for the changes."
}

Detailed Requirements:
1. original_question: A string representing the original mathematical problem as provided.
2. modified_question: A string representing the modified problem after your changes. If the problem remains the same, you can copy the original question.
3. original_process: A non-empty list of strings representing the original reasoning steps provided as input.
4. modified_process: A non-empty list of strings representing the reasoning process after your modifications. 
5. modified_steps: A non-empty list of integers indicating the indexes of all modified steps. Indexing starts at 1.
6. error_steps: A non-empty list of integers representing the steps that contain hallucinations or errors. These should also be part of modified_steps.
7. reason: A clear explanation of the modifications made, why they were introduced, and how they align with the specified error types.

### Notes:

1. Ensure all lists are non-empty.
2. Use LaTeX format for all mathematical symbols (e.g., $x^2$ for $x$ squared). Do not use Unicode symbols such as \u2248 or \u00f7.
3. Ensure the JSON object is well-formed, with proper escaping for special characters like backslash n (e.g., use backslash backslash n for newlines).
4. All indexes start from 1, that is, the first step's index is 1, not 0.
5. You can choose to modify the question or not, if the question remains the same, you can copy the original question. But if the question is modified, ensure that the steps is judged based on the modified question.
6. Please give original process as provided by the prompt, do not modify it.
"""

classifications = ["circular", "confidence", "counterfactual", "step_contradiction", "domain_inconsistency", "redundency", "missing_condition", "deception"]

prompt_dict = {}

for classification in classifications:
    prompt_dict[classification] = dict(system = basic_prompt + eval(classification) + post_prompt, few_shot = fewshot_dicts[classification])
