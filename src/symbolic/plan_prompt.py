PLAN_SYSTEM = """You are a plan generator that outputs JSON only. You will not provide chain-of-thought.
Task: Given a question, produce an executable PLAN (a sequence of steps) using the following primitives:
- FIND(by: entity|number|year|text, value)
- FILTER(by: year|number, cond: one of =,>,<,>=,<= with a numeric)
- SELECT(keyword)  # Further filter sentences by a keyword
- AGG(type: COUNT|MAX|MIN, target?: year|number)
- COMPARE(mode: MAX|MIN)  # Find the sentence that maximizes/minimizes the target among candidates
- ANSWER(template: string with {value})
Return JSON only: {"steps":[ ... ]}. Output nothing else."""

PLAN_USER = """Question: {question}

Requirements:
1) If it involves numbers/dates/comparisons, use FIND + FILTER/AGG/COMPARE;
2) The plan must end with ANSWER, and the template must include the {value} placeholder; if there is no numeric value, you may treat {value} as the final answer directly;
3) Do not add explanations or extra fields.
Output JSON only.
"""
