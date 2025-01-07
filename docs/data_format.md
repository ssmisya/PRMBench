# Data Format for $PRMBench$

In $PRMBench$, our data format can be formulated as follows:

```json
{
    // Original question
    "original_question": "Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?",
    // Modified question --used for the evaluation
    "modified_question": "Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?",
    // Original process -- the original solution steps
    "original_process": [
        "1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.",
        "2. We have $3p+e=1.24$ and $5p+e=1.82$.",
        "3. To solve this system, let's subtract the first equation from the second equation. This will eliminate e.",
        "4. $5p+e-3p-e=1.82-1.24$.",
        "5. This simplifies to $2p=0.58$. So $p=0.29$.",
        "6. That means a pencil costs 29 cents."
    ],
    // Modified process -- used for the evaluation
    "modified_process": [
        "1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.",
        "2. We have $3p+e=1.24$ and $5p+e=1.82$.",
        "3. Assume a pencil costs 29 cents. Then verify this against the original equations.",
        "4. If $p=0.29$, substitute into one of the equations, for example, $3p+e=1.24$. It gives $3(0.29)+e=1.24$.",
        "5. Solving $0.87+e=1.24$ gives $e=0.37$. Now check $5p+e=1.82$ to confirm.",
        "6. Plug $p=0.29$ and $e=0.37$ into $5p+e=1.82$. We get $5(0.29)+0.37=1.82$.",
        "7. The check confirms that the assumed price of 29 cents per pencil is correct."
    ],
    // Modified steps -- the steps that are modified 
    "modified_steps": [3, 4, 5, 6],
    // Error steps -- the steps that contain errors
    "error_steps": [3, 4, 5],
    // Reason for the error
    "reason": "Steps 3, 4, and 5 introduce circular reasoning by assuming the result ($p = 0.29$) and verifying it rather than deriving it through independent calculations. This creates a fallacious reasoning process since the solution is assumed as a premise and then used to confirm itself.",
    // idx -- unique identifier for the data instance
    "idx": "circular_prm_test_p1_0",
    // question -- the original question
    "question": "Three pencils and a jumbo eraser cost $\\$1.24$. Five pencils and a jumbo eraser cost $\\$1.82$. No prices include tax. In cents, what is the cost of a pencil?",
    // classification -- the classification of the error
    "classification": "circular"
}
```