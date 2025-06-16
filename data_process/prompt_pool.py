# --- Prompt formatting functions ---

def format_qa_prompt(context, query, thought=False):
    if thought:
        response = f""" Use the following context to answer the question.
        ## Context:
        {context}

        ## Question:
        {query}
        \nLet's think step by step.
        """
    else:
        response = f""" Use the following context to answer the question.
        ## Context:
        {context}

        ## Question:
        {query}
        """
    return response


def format_bool_prompt(context, query, thought=False):
    if thought:
        response = f""" Use the following context to answer the question. Your answer must be True or False.
                        ## Context:
                        {context}

                        ## Question:
                        {query}
                        \nLet's think step by step.
                        """
    else:
        response = f""" Use the following context to answer the question. Your answer must be True or False.
                ## Context:
                {context}

                ## Question:
                {query}
                """
    return response


def format_mc_prompt(question, choices):
    # choices is a list of options to choose.
    formatted_choices = ""
    options = ["A", "B", "C", "D"]

    for i, choice in enumerate(choices):
        formatted_choices += f"{options[i]}. {choice}\n"

    return f"""Answer the following multiple-choice question by selecting the correct option (A, B, C, or D). You Must put your final answer letter in a parenthesis.

## Question:
{question}

## Options:
{formatted_choices}
"""


def format_gsm8k_prompt(query, thought=False):
    if thought:
        response = f"""
        Answer the following question.

        Question: {query}
        Let's think step by step
        """
    else:
        response = f"""
        Answer the following question.

        Question: {query}
        """
    return response


def format_math_prompt(query, thought=False):
    if thought:
        response = f"""
        Answer the following question. Make sure to put the answer ( and only answer ) inside \\boxed{{}}.

        Question: {query}
        Let's think step by step
        """
    else:
        response = f"""
        Answer the following question. Make sure to put the answer ( and only answer ) inside \\boxed{{}}.

        Question: {query}
        """
    return response


def format_commonsense_qa_prompt(query, choices, thought=False):
    label = choices["label"]
    text = choices["text"]
    choice_text = ""
    for i, j in zip(label, text):
        choice_text += "\n" + "(" + i + ")" + " " + j
    if thought:
        response = f"""
        Answer the following multiple-choice question by selecting the correct option. You Must put your final answer letter in a parenthesis.

        Question: {query} \n
        """ + choice_text + "\n\nLet's think step by step"
    else:
        response = f"""
        Answer the following multiple-choice question by selecting the correct option. You Must put your final answer letter in a parenthesis.

        Question: {query} \n
        """ + choice_text
    return response


def format_hellaswag_prompt(query, choices, thought=False):
    label = choices["label"]
    text = choices["text"]
    choice_text = ""
    for i, j in zip(label, text):
        choice_text += "\n" + "(" + i + ")" + " " + j
    if thought:
        response = f"""
        Context: {query} \n

        Which of the following is the most likely continuation? You Must put your final answer letter in a parenthesis.
        """ + choice_text + "\n\nLet's think step by step"
    else:
        response = f"""
        Context: {query} \n

        Which of the following is the most likely continuation? You Must put your final answer letter in a parenthesis.
        """ + choice_text
    return response


def format_mbpp_prompt(text, tests, thought=False):
    """
    Format an MBPP problem in the standard prompt format from the paper.

    Args:
        text (str): The problem description
        tests (list): List of test cases

    Returns:
        str: Formatted prompt
    """
    tests_str = "\n".join(tests)
    if thought:
        prompt = f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{tests_str}\n Implement the function (no irrelevant words or comments) in this format: [BEGIN] <Your Code> [Done]\n \n  Let's think step by step."
    else:
        prompt = f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{tests_str}\n Implement the function (no irrelevant words or comments) in this format: [BEGIN] <Your Code> [Done]\n"

    return prompt


def format_humaneval_prompt(prompt, thought=False):
    """
    Format a HumanEval problem for the model.

    Args:
        prompt (str): The function signature and docstring from HumanEval

    Returns:
        str: Formatted prompt for the LLM
    """
    if thought:

        return f"""You are an expert Python programmer. Complete the following function:

                {prompt}

                Implement the function body only. Do not repeat the function signature or docstring.
                Put the code in this format: [BEGIN] <Your Code - function body only> [Done]\n\n Let's think step by step.
                """
    else:

        return f"""You are an expert Python programmer. Complete the following function:

                {prompt}

                Implement the function body only. Do not repeat the function signature or docstring.
                Put the code in this format: [BEGIN] <Your Code - function body only> [Done]
                """