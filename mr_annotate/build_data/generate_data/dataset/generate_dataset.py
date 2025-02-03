
from torch.utils.data import Dataset
from mr_eval.utils.utils import *
# Question:

# If $x^2 - x - 1 = 0$, what is the value of $x^3 - 2x + 1$?

# Answer:

# Step 1. Let's start with the first equation and see if we can solve for x.

# Step 2. We can use the quadratic formula to find x.

# Step 3. The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

# Step 4. Substituting $a = 1$, $b = -1$, and $c = -1$, we get $x = \\frac{1 \\pm \\sqrt{5}}{2}$.

# Step 5. Since $x^2 - x - 1 = 0$, we replace $x^2$ with $x + 1$ in $x^3 - 2x + 1$.

# Step 6. Substituting, $x^3 - 2x + 1 = x(x + 1) - 2x + 1$.

# Step 7. Simplify: $x^3 - 2x + 1 = (x^2 + x) - 2x + 1$.

# Step 8. Substitute $x^2 = x + 1$: $x^3 - 2x + 1 = (x + 1 + x) - 2x + 1$.

# Step 9. Simplify: $x^3 - 2x + 1 = 2$."

# # Answer

# 2

class GenerateDataset(Dataset):
    def __init__(self, data_args):
        self.data_args = data_args 
        self.input_path = data_args.input_path
        self.output_path = data_args.output_path 
        self.subject = data_args.subject
        
        self.load_data()
        self.resume_from_ckpt()
        
    def load_data(self):
        raw_data = process_jsonl(self.input_path)
        self.meta_data = []
        for idx, item in enumerate(raw_data):
            item_idx = item["idx"]
            question = item["Question"]
            options = item["Options"]
            question = f"{question} {options}"
            steps = item["Model_Solution_Steps"]
            step_str = ""
            for step_idx,step in steps:
                step_text = step["text"]
                question = f"{question} {step_text}"
            
            
    
    def resume_from_ckpt(self):
        pass
    
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        pass