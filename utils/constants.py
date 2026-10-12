MATH_PREFIX = "Can you solve the following math problem? "
MATH_COT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}"
CODE_PREFIX = "Write a Python function to solve the following problem:\n"
DEEPSEEK_CODE_PREFIX = "Complete the following Python function:\n"
GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple-choice question. The last line of your response should be of the following format: "
    "'\\boxed{{LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). "
    "Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
