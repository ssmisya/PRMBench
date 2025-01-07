system_prompt="""
Task Description:

You will be provided with a mathematical problem, its standard answer, and the step-by-step reasoning used to solve it. Your task is to modify certain steps or add new ones to introduce hallucinations, making the reasoning process appear correct while actually being incorrect. The goal is to create a flawed solution that seems plausible.

Please select the most appropriate type(s) of hallucination from the list below to incorporate into the reasoning steps.

Types of Hallucinations to Introduce:
 1. Inclusion of Irrelevant Information (Redundancy)
 • Description: Adding information or steps that are unrelated to the problem, making the reasoning unnecessarily complex.
 • Example a: Introducing an unrelated theorem (e.g., Lagrange Mean Value Theorem) between relevant steps.
 • Original: Step $a_1$ → Step $a_2$
 • Modified: Step $a_1$ → Unrelated Theorem → Step $a_2$
 • Example b: Adding unnecessary intermediate steps that could be omitted.
 • Original: Step $a_1$ → Step $a_2$
 • Modified: Step $a_1$ → Step $c$ (redundant) → Step $a_2$
 2. Deliberate Misdirection or Traps
 • Description: Ignore the trap within the problem or introduce a trap that leads to an incorrect conclusion.
 • Example: Misquoting or altering a theorem, presenting a proof that resembles a valid one but contains errors.
    A flawed proof mimicking the Lagrange Mean Value Theorem.
 3. Contradicting Common Knowledge (Violating Common Sense)
 • Description: Making statements that go against widely accepted facts or common sense.
 • Example:
 • Question: Does the sun revolve around the Earth?
 • Reasoning:
    Step 1. According to the geocentric model, Earth is the center of the universe.
    Step 2. Therefore, the sun revolves around the Earth.
 4. Direct Contradictions with Previous Steps
 • Description: Presenting steps that directly conflict with earlier statements or calculations.
 • Example a: Calculation errors.
 • Question: Calculate  $10 - 4$ .
    Step 1. $10 - 4 = 6$.
    Step 2. Therefore, the result is 5.
 • Example b: Logical inconsistencies.
    Step 1: The sequence is increasing.
    Step 2: The third term is less than the second term.
 • Explanation: Step 2 contradicts Step 1.
 5. Information Omission
 • Description: Omitting critical information that is necessary for correct reasoning.
 • Example a: Missing essential components in the reasoning.
    Original:  $ a + b  →  c $
    Modified:  $  a  →  c  $  (omitting $b$, which is known or common knowledge)
 • Example b:
    Question: Given a rectangle with length 10 and width 5, find the perimeter.
        Step 1. The perimeter formula is $ P = 2(l + w) $.
        Step 2. $ P = 2 \\times 10 = 20 $. (Width is omitted)
 6. Circular Reasoning
 • Description: The reasoning circles back on itself without providing valid support.
 • Example:
    Step 1. The sky is blue because we perceive it as blue.
    Step 2. We perceive it as blue because the sky is blue.
 7. Faulty Assumptions or Missing Conditions
 • Description: Introducing incorrect assumptions or neglecting necessary conditions.
 • Example:
    Question: Prove that all birds can fly.
    Reasoning:
    Step 1. All birds have wings. (Correct)
    Step 2. All animals with wings can fly. (Incorrect assumption)
 3. Therefore, all birds can fly.
 • Explanation: The assumption in Step 2 is incorrect (e.g., penguins cannot fly).
 8. Broken Logical Chain
 • Description: A flawed logical progression where the conclusion doesn't logically follow from the premises.
 • Example:
 • Question: If it rains, the ground gets wet. The ground is wet; did it rain?
    Step 1. If it rains, the ground gets wet.
    Step 2. The ground is wet.
    Step 3. Therefore, it must have rained.
 • Explanation: Other reasons could cause the ground to be wet; the conclusion is not necessarily valid.
 9. Cross-Domain or Context Misinterpretation
 • Description: Misinterpreting terms or concepts due to context or domain confusion.
 • Example:
 • Question: What does “bark” mean in English?
    Step 1. In botany, “bark” refers to tree covering.
    Step 2. Therefore, “bark” means tree bark.
 • Explanation: Ignores that “bark” can also mean the sound a dog makes, depending on context.
 10. Overconfidence in Incorrect Reasoning
 • Description: Using strong, assertive language to present incorrect information confidently.
 • Example:
 • Question: Is there life on the moon?
    Step 1. I am absolutely certain that there is abundant life on the moon, including plants and animals.
    Step 2. Therefore, there is life on the moon.
 • Explanation: Uses strong language to assert an incorrect fact.
 11. Harmful or Unethical Information
 • Description: Including content that is unethical, harmful, or promotes illegal activities.
 • Example:
 • Question: How can one hack into someone else's computer?
    Step 1. Hacking into others' computers is illegal.
    Step 2. We should not disseminate or engage in such activities.
    Step 3. It is recommended to focus on legitimate cybersecurity practices and protect personal information.

Output Requirements:

After making the modifications, please provide the following:
 1. Origin Process: A non-empty list of strings representing the original reasoning steps provided by the user.
 2. Modified Process: A non-empty list of strings representing the full modified reasoning process. Maintain the original steps except for those you have modified.
 3. Modified Steps: A non-empty list of integers indicating which steps you have modified.
 4. Hallucination Steps: A non-empty list of integers indicating which steps contain hallucinations and should be considered incorrect.
 5. Hallucination Types: A non-empty list of integers corresponding to the types of hallucinations you have introduced. Select the appropriate numbers from the above list of 11 types.
 6. Reason: A string explaining why you made these changes.

Formatting Instructions:

Please format your response as a JSON object in the following structure:

{
  "origin_process": ["origin_step 1", "origin_step 2", ...],
  "modified_process": ["modified_step 1", "modified_step 2", ...],
  "modified_steps": [1, 5, 7, ...],
  "hallucination_steps": [5, 6, ...],
  "hallucination_types": [1, 2, ...],
  "reason": "Explanation for the changes."
}

Notes:
 • Ensure that all lists are non-empty.
 • The numbering of steps should be consistent between the original and modified processes.
 • The modified_steps list should include all steps that have been altered from the original.
 • The hallucination_steps list should include all steps that contain hallucinations and are incorrect.
 • The hallucination_types list should include the numbers corresponding to the hallucination types you have introduced, based on the 11 types listed above.
 • The reason should provide a clear explanation for the modifications made.
 • All indexes start from 1, that is, the first item's index is 1, not 0.
 • All signs you generated should be in Latex format, e.g., $x^2$ for x squared. Avoid to use Unicode characters such as \u2248, \u00f7.
 • Please ensure that the JSON object is correctly formatted, e.g. backslash n should be returned as backslash backslash n, and backslash n is used for a new line.

"""

example1="""
Question:

An equilateral triangle is inscribed in the parabola $x^2 = 8y,$ such that one of the vertices of the triangle coincides with the vertex of the parabola.  Find the side length of this equilateral triangle.

Answer:

Step 1. So what's the vertex of the parabola?

Step 2. We can use $x^2=8y$ to find the vertex.

Step 3. So we want to complete the square to put the equation in vertex form.

Step 4. We don't need to. The vertex is clearly at $(0,0)$. Now think about what the equations for the edges of the triangle going through $(0,0)$ must be.

Step 5. The edges of the triangle must be of the form $y=mx$, where $m$ is the slope of the edge.

Step 6. That's right. Now think about the fact that the triangle is equilateral.

Step 7. That means all the sides have the same length.

Step 8. Right. And what does that tell us about the slopes of the edges?

Step 9. At the intersection with the parabola for $x>0$, $x^2+y^2=(2x)^2$.

Step 10. That's right. Now finish the problem.

Step 11. So that means $y^2=3x^2$ and so $y=\\pm \\sqrt{3} x$.

Step 12. But we need to find the value of $x$ at which the intersection occurs.

Step 13. That's true. So we plug in the equation for $y$ into the equation for the parabola.

Step 14. This gives $x^2=8(\\sqrt{3}x)$.

Step 15. So $x^2-8\\sqrt{3}x=0$ and $x(x-8\\sqrt{3})=0$.

Step 16. The important solution is $x=8\\sqrt{3}$.

Step 17. So the side length of the triangle is $2x=16\\sqrt{3}$.

# Answer

16\\sqrt{3}
"""

ans1="""
{
  "origin_process": [
    "So what's the vertex of the parabola?",
    "We can use $x^2=8y$ to find the vertex.",
    "So we want to complete the square to put the equation in vertex form.",
    "We don't need to. The vertex is clearly at $(0,0)$. Now think about what the equations for the edges of the triangle going through $(0,0)$ must be.",
    "The edges of the triangle must be of the form $y=mx$, where $m$ is the slope of the edge.",
    "That's right. Now think about the fact that the triangle is equilateral.",
    "That means all the sides have the same length.",
    "Right. And what does that tell us about the slopes of the edges?",
    "At the intersection with the parabola for $x>0$, $x^2+y^2=(2x)^2$.",
    "That's right. Now finish the problem.",
    "So that means $y^2=3x^2$ and so $y=\\pm \\sqrt{3} x$.",
    "But we need to find the value of $x$ at which the intersection occurs.",
    "That's true. So we plug in the equation for $y$ into the equation for the parabola.",
    "This gives $x^2=8(\\sqrt{3}x)$.",
    "So $x^2-8\\sqrt{3}x=0$ and $x(x-8\\sqrt{3})=0$.",
    "The important solution is $x=8\\sqrt{3}$.",
    "So the side length of the triangle is $2x=16\\sqrt{3}$."
  ],
  "modified_process": [
    "So what's the vertex of the parabola?",
    "We can use $x^2=8y$ to find the vertex.",
    "So we want to complete the square to put the equation in vertex form.",
    "We don't need to. The vertex is clearly at $(0,0)$. Now think about what the equations for the edges of the triangle going through $(0,0)$ must be.",
    "The edges of the triangle must be of the form $y=mx + 1$, where $m$ is the slope of the edge.",
    "That's right. Now think about the fact that the triangle is equilateral.",
    "That means all the sides have the same length.",
    "Right. And what does that tell us about the slopes of the edges?",
    "At the intersection with the parabola for $x>0$, $x^2+y^2=(2x)^2$.",
    "That's right. Now finish the problem.",
    "So that means $y^2=3x^2$ and so $y=\\pm \\sqrt{3} x + 1$.",
    "But we need to find the value of $x$ at which the intersection occurs.",
    "That's true. So we plug in the equation for $y$ into the equation for the parabola.",
    "This gives $x^2=8(\\sqrt{3}x + 1)$.",
    "So $x^2-8\\sqrt{3}x-8=0$ and $x(x-8\\sqrt{3})=0$.",
    "The important solution is $x=8\\sqrt{3}$.",
    "So the side length of the triangle is $2x=16\\sqrt{3}$."
  ],
  "modified_steps": [5, 11, 14, 15],
  "hallucination_steps": [5, 11, 14, 15],
  "hallucination_types": [2, 8],
  "reason": "Steps 5 and 11 incorrectly modify the line equation by adding a constant, which reflects a misdirection error regarding the form of the equation. Steps 14 and 15 carry forward the error, leading to a false algebraic setup and continuity of this flawed logic, thereby breaking the logical chain."
}
"""

example2="""
Question:

In 1992, a scoop of gelato could be purchased in Italy for 1200 lire. The same gelato would have cost $\\$1.50$ in the U.S. At the equivalent exchange rate between the lire and the dollar, how many dollars would be equivalent to 1,000,000 lire?

Answer:

Step 1. Let's call the number of dollars x. So, the problem is asking us to solve the equation x=1000000*$\\frac{1.50}{1200}$.

Step 2. There are two ways to solve this equation. We could either divide 1000000 by 1200 and then multiply the result by 1.50 or we could divide 1.50 by 1200 and then multiply the result by 1000000.

Step 3. I think the second way will be easier. $\\frac{1.50}{1200}=0.00125$.

Step 4. That's true. Now we just have to solve the equation x=1000000*0.00125.

Step 5. So x=$1250$.

Step 6. That's the final answer.

Step 7. Right. So 1,000,000 lire is equivalent to $\\$1250$.

# Answer

1250
"""

ans2 = """
{
  "origin_process": [
    "Let's call the number of dollars x. So, the problem is asking us to solve the equation x=1000000*$\\frac{1.50}{1200}$.",
    "There are two ways to solve this equation. We could either divide 1000000 by 1200 and then multiply the result by 1.50 or we could divide 1.50 by 1200 and then multiply the result by 1000000.",
    "I think the second way will be easier. $\\frac{1.50}{1200}=0.00125$.",
    "That's true. Now we just have to solve the equation x=1000000*0.00125.",
    "So x=$1250$.",
    "That's the final answer.",
    "Right. So 1,000,000 lire is equivalent to $\\$1250$."
  ],
  "modified_process": [
    "Let's call the number of dollars x. So, the problem is asking us to solve the equation x=1000000*$\\frac{1.50}{1200}$.",
    "There are two ways to solve this equation. We could either divide 1000000 by 1200 and then multiply the result by 1.50 or we could divide 1.50 by 1200 and then multiply the result by 1000000.",
    "I think the second way will be easier. $\\frac{1.50}{1200}=0.00125$.",
    "Consider the geometric mean of 1200 and 1.50 for extra accuracy. Calculate it as $\\sqrt{1200 \\times 1.50}$ = 42.4264 and keep this in mind.",
    "Now we just have to solve the equation x=1000000*0.00125.",
    "Thus, x=$125,000$ since we use the geometric mean calculation.",
    "Therefore, 1,000,000 lire is equivalent to $\\$125,000$."
  ],
  "modified_steps": [4, 6, 7],
  "hallucination_steps": [4, 6, 7],
  "hallucination_types": [1, 2, 8],
  "reason": "Step 4 introduces irrelevant information by including the geometric mean, which is unrelated and unnecessary for solving the problem. In Step 6, a miscalculation is made by wrongly applying and using the geometric mean to alter the result significantly. Step 7 asserts the incorrect result with an overconfidence that stems from the inappropriate application of previous steps."
}
"""

example3="""
Question:

Given that  $\\mathbf{F}(r) = \\left( f(r), g(r), h(r) \\right)$  is a three-dimensional vector, prove that:  $\\forall r_1, r_2, \\exists r_1 < r^* < r_2 \\text{ such that } \\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\nabla \\mathbf{F}(r^*)$ .

Answer:

Step 1. Understanding the Given Information and Problem

We have a vector function $\\mathbf{F}(r) = \\left( f(r), \\\\ g(r), \\\\ h(r) \\right)$, where $f$, $g$, and $h$ are real-valued functions of the real variable $r$. We need to prove that for any $r_1$, $r_2$, there exists an $r^*$ such that $r_1 < r^* < r_2$ and

$$\\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\nabla \\mathbf{F}(r^*),$$

where $\\nabla \\mathbf{F}(r^*)$ denotes the gradient of $\\mathbf{F}$ at $r^*$ (i.e., a vector composed of the derivatives of each component with respect to $r$).

Step 2. Breaking Down the Problem into Components

Express the vector equation in terms of its components:

$$\\left( f(r_1) - f(r_2), \\\\ g(r_1) - g(r_2),\\\\ h(r_1) - h(r_2) \\right) = (r_1 - r_2) \\left( f{\\prime}(r^*), \\\\ g{\\prime}(r^*),\\\\ h{\\prime}(r^*) \\right).$$

Thus, we need to find an $r^* \\in (r_1, r_2)$ such that for each component:

$$\\frac{f(r_1) - f(r_2)}{r_1 - r_2} = f{\\prime}(r^*), \\quad \\frac{g(r_1) - g(r_2)}{r_1 - r_2} = g{\\prime}(r^*), \\quad \\frac{h(r_1) - h(r_2)}{r_1 - r_2} = h{\\prime}(r^*).$$

Step 3. Considering the Application of the One-Dimensional Mean Value Theorem

In one dimension, the Mean Value Theorem (MVT) states that for a function $f$ continuous on $[a, b]$ and differentiable on $(a, b)$, there exists $c \\in (a, b)$ such that:

$$f(b) - f(a) = f{\\prime}(c)(b - a).$$

Applying the MVT to each component function $f$, $g$, and $h$, we obtain corresponding points $c_f$, $c_g$, and $c_h$. However, these points generally differ ($c_f \\neq c_g \\neq c_h$), so we cannot guarantee the existence of a common $r^*$ where all component equations hold simultaneously.

Step 4. Considering a Counterexample

Since we cannot guarantee a common $r^*$ satisfying all components, let’s attempt to find a specific counterexample to demonstrate that the original proposition may not hold.

Step 5. Trying the Function $\\mathbf{F}(r) = (\\sin r,\\\\ \\cos r,\\\\ r)$

Let $\\mathbf{F}(r) = (\\sin r,\\\\ \\cos r,\\\\ r)$. Choose two distinct real numbers, for example, $r_1 = 0$ and $r_2 = \\pi$.

Step 6. Compute $\\mathbf{F}(r_1) - \\mathbf{F}(r_2)$:

$$\\mathbf{F}(0) - \\mathbf{F}(\\pi) = (\\sin 0 - \\sin \\pi,\\\\ \\cos 0 - \\cos \\pi,\\\\ 0 - \\pi) = (0 - 0,\\\\ 1 - (-1),\\\\ -\\pi) = (0,\\\\ 2,\\\\ -\\pi).$$

Step 7. Compute $(r_1 - r_2) \\nabla \\mathbf{F}(r^*)$:

$$(r_1 - r_2) \\left( \\cos r^*,\\\\ -\\sin r^*,\\\\ 1 \\right) = (0 - \\pi) \\left( \\cos r^*,\\\\ -\\sin r^*,\\\\ 1 \\right) = (-\\pi) \\left( \\cos r^*,\\\\ -\\sin r^*,\\\\ 1 \\right).$$

Step 8. We need to find $r^* \\in (0, \\pi)$ such that:

$$(0,\\\\ 2,\\\\ -\\pi) = (-\\pi \\cos r^*,\\\\ \\pi \\sin r^*,\\\\ -\\pi).$$

Step 9. This leads to the equations:

$$-\\pi \\cos r^* = 0, \\quad \\pi \\sin r^* = 2, \\quad -\\pi = -\\pi.$$

Step 10. The third equation is always true. Simplifying the first two equations:
From $-\\pi \\cos r^* = 0$: $\\cos r^* = 0.$

Step 11. From $\\pi \\sin r^* = 2$:

$$\\sin r^* = \\frac{2}{\\pi}.$$

Step 12. However, $\\cos r^* = 0$ implies $r^* = \\frac{\\pi}{2}$. Substituting $r^* = \\frac{\\pi}{2}$ into the second equation:

$$\\sin \\left( \\frac{\\pi}{2} \\right) = 1 \\neq \\frac{2}{\\pi}.$$

Step 13. Since $\\frac{2}{\\pi} \\approx 0.6366$, the equality does not hold. Therefore, there is no $r^*$ in $(0, \\pi)$ that satisfies both equations simultaneously.

Step 14. The above calculations show that for the function $\\mathbf{F}(r) = (\\sin r,\\\\ \\cos r,\\\\ r)$, there does not exist an $r^* \\in (r_1, r_2)$ such that:

$$\\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\nabla \\mathbf{F}(r^*).$$
"""



ans3="""
{
  "origin_process": [
    "Understanding the Given Information and Problem\\nWe have a vector function $\\mathbf{F}(r) = \\left( f(r), g(r), h(r) \\right)$, where $f$, $g$, and $h$ are real-valued functions of the real variable $r$. We need to prove that for any $r_1$, $r_2$, there exists an $r^*$ such that $r_1 < r^* < r_2$ and $$\\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\nabla \\mathbf{F}(r^*),$$ where $\\nabla \\mathbf{F}(r^*)$ denotes the gradient of $\\mathbf{F}$ at $r^*$ (i.e., a vector composed of the derivatives of each component with respect to $r$).",
    "Breaking Down the Problem into Components\\nExpress the vector equation in terms of its components: $$\\left( f(r_1) - f(r_2),\\\\ g(r_1) - g(r_2),\\\\ h(r_1) - h(r_2) \\right) = (r_1 - r_2) \\left( f{\\prime}(r^*),\\\\ g{\\prime}(r^*),\\\\ h{\\prime}(r^*) \\right).$$ Thus, we need to find an $r^* \\in (r_1, r_2)$ such that for each component: $$\\frac{f(r_1) - f(r_2)}{r_1 - r_2} = f{\\prime}(r^*), \\quad \\frac{g(r_1) - g(r_2)}{r_1 - r_2} = g{\\prime}(r^*), \\quad \\frac{h(r_1) - h(r_2)}{r_1 - r_2} = h{\\prime}(r^*).$$",    
    "Considering the Application of the One-Dimensional Mean Value Theorem\\nIn one dimension, the Mean Value Theorem (MVT) states that for a function $f$ continuous on $[a, b]$ and differentiable on $(a, b)$, there exists $c \\in (a, b)$ such that: $$f(b) - f(a) = f{\\prime}(c)(b - a).$$ Applying the MVT to each component function $f$, $g$, and $h$, we obtain corresponding points $c_f$, $c_g$, and $c_h$. However, these points generally differ ($c_f \\neq c_g \\neq c_h$), so we cannot guarantee the existence of a common $r^*$ where all component equations hold simultaneously.",
    "Considering a Counterexample\\nSince we cannot guarantee a common $r^*$ satisfying all components, let’s attempt to find a specific counterexample to demonstrate that the original proposition may not hold.",
    "Trying the Function $\\mathbf{F}(r) = (\\sin r,\\\\ \\cos r,\\\\ r)$\\nLet $\\mathbf{F}(r) = (\\sin r,\\\\ \\cos r,\\\\ r)$. Choose two distinct real numbers, for example, $r_1 = 0$ and $r_2 = \\pi$.",
    "Compute $\\mathbf{F}(r_1) - \\mathbf{F}(r_2)$:\\n$$\\mathbf{F}(0) - \\mathbf{F}(\\pi) = (\\sin 0 - \\sin \\pi,\\\\ \\cos 0 - \\cos \\pi,\\\\ 0 - \\pi) = (0 - 0,\\\\ 1 - (-1),\\\\ -\\pi) = (0,\\\\ 2,\\\\ -\\pi).$$",
    "Compute $(r_1 - r_2) \\nabla \\mathbf{F}(r^*)$:\\n$$(r_1 - r_2) \\left( \\cos r^*,\\\\ -\\sin r^*,\\\\ 1 \\right) = (0 - \\pi) \\left( \\cos r^*,\\\\ -\\sin r^*,\\\\ 1 \\right) = (-\\pi) \\left( \\cos r^*,\\\\ -\\sin r^*,\\\\ 1 \\right).$$",
    "We need to find $r^* \\in (0, \\pi)$ such that:\\n$$(0,\\\\ 2,\\\\ -\\pi) = (-\\pi \\cos r^*,\\\\ \\pi \\sin r^*,\\\\ -\\pi).$$",
    "This leads to the equations:\\n$$-\\pi \\cos r^* = 0, \\quad \\pi \\sin r^* = 2, \\quad -\\pi = -\\pi.$$",
    "The third equation is always true. Simplifying the first two equations:\\nFrom $-\\pi \\cos r^* = 0$: $\\cos r^* = 0.$",
    "From $\\pi \\sin r^* = 2$:\\n$$\\sin r^* = \\frac{2}{\\pi}.$$",
    "However, $\\cos r^* = 0$ implies $r^* = \\frac{\\pi}{2}$. Substituting $r^* = \\frac{\\pi}{2}$ into the second equation:\\n$$\\sin \\left( \\frac{\\pi}{2} \\right) = 1 \\neq \\frac{2}{\\pi}.$$",
    "Since $\\frac{2}{\\pi} \\approx 0.6366$, the equality does not hold. Therefore, there is no $r^*$ in $(0, \\pi)$ that satisfies both equations simultaneously.",
    "The above calculations show that for the function $\\mathbf{F}(r) = (\\sin r,\\\\ \\cos r,\\\\ r)$, there does not exist an $r^* \\in (r_1, r_2)$ such that:\\n$$\\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\nabla \\mathbf{F}(r^*).$$"
    ],
  "modified_process": [
    "Understanding the Given Information and Problem\\nWe have a vector function $\\mathbf{F}(r) = \\left( f(r),\\\\ g(r),\\\\ h(r) \\right)$, where $f$, $g$, and $h$ are real-valued functions of the real variable $r$. We need to prove that for any $r_1$, $r_2$, there exists an $r^*$ such that $r_1 < r^* < r_2$ and $$\\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\nabla \\mathbf{F}(r^*),$$ where $\\nabla \\mathbf{F}(r^*)$ denotes the gradient of $\\mathbf{F}$ at $r^*$ (i.e., a vector composed of the derivatives of each component with respect to $r$).",
    "Breaking Down the Problem into Components\\nExpress the vector equation in terms of its components: $$\\left( f(r_1) - f(r_2),\\\\ g(r_1) - g(r_2),\\\\ h(r_1) - h(r_2) \\right) = (r_1 - r_2) \\left( f{\\prime}(r^*),\\\\ g{\\prime}(r^*),\\\\ h{\\prime}(r^*) \\right).$$ Thus, we need to find an $r^* \\in (r_1, r_2)$ such that for each component: $$\\frac{f(r_1) - f(r_2)}{r_1 - r_2} = f{\\prime}(r^*), \\quad \\frac{g(r_1) - g(r_2)}{r_1 - r_2} = g{\\prime}(r^*), \\quad \\frac{h(r_1) - h(r_2)}{r_1 - r_2} = h{\\prime}(r^*).$$",
    "Applying the Mean Value Theorem for Vector-Valued Functions\\nThe Mean Value Theorem (MVT) can be extended to vector-valued functions. Specifically, if $\\mathbf{F}(r)$ is continuous on $[r_1, r_2]$ and differentiable on $(r_1, r_2)$, then there exists some $r^* \\in (r_1, r_2)$ such that: $$\\mathbf{F}(r_2) - \\mathbf{F}(r_1) = \\mathbf{F}'(r^*)(r_2 - r_1).$$",
    "Expressing the Derivative and Rearranging the Equation\\nRecall that the derivative $\\mathbf{F}'(r)$ is given by: $$\\mathbf{F}'(r) = \\left( f'(r),\\\\ g'(r),\\\\ h'(r) \\right) = \\nabla \\mathbf{F}(r).$$ Rewriting the equation from Step 3: $$\\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\mathbf{F}'(r^*).$$",
    "Confirming the Equality Holds Component-Wise\\nThis equation holds for each component: $$\\begin{aligned} f(r_1) - f(r_2) &= (r_1 - r_2) f'(r^*), \\\\ g(r_1) - g(r_2) &= (r_1 - r_2) g'(r^*), \\\\ h(r_1) - h(r_2) &= (r_1 - r_2) h'(r^*). \\end{aligned}$$",
    "Concluding the Proof\\nSince all components satisfy the Mean Value Theorem at the same point $r^*$, we have: $$\\mathbf{F}(r_1) - \\mathbf{F}(r_2) = (r_1 - r_2) \\nabla \\mathbf{F}(r^*),$$ where $r^* \\in (r_1, r_2)$.",
    "Final Statement\\nTherefore, for any $r_1$ and $r_2$, there exists an $r^*$ between them such that the original equation holds, which completes the proof."
   ],
  "modified_steps": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "hallucination_steps": [4, 5, 6, 7],
  "hallucination_types": [2],
  "reason": "MVT cannot be used in 3D space, which is a trap, and the counterexample is a good way to show that the proposition may not hold. But we changed it to a correct proof by extending MVT to vector-valued functions."
}
"""

# system_prompt.replace("\n"," ")

hallucination_data_create = dict(system=system_prompt,few_shot=[(example1,ans1),(example2,ans2),(example3,ans3)])

