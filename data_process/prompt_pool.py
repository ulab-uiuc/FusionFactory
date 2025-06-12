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


def format_quac_prompt(
        sample: dict,
        turn_idx: int = 0,
        thought: bool = False
) -> str:
    """
    Build a QuAC-style prompt for a single turn.

    Parameters
    ----------
    sample : dict
        One element from the QuAC dataset.
    turn_idx : int, default 0
        Target turn in this dialogue.
    thought : bool, default False
        If True, append 'Let's think step by step.' before the answer box.

    Returns
    -------
    str
        Prompt string ready for the LLM.
    """
    # 1. Header information
    page_title = sample["wikipedia_page_title"]
    background = sample["background"]
    section = sample["section_title"]
    context = sample["context"]

    # 2. Dialogue history (before current turn)
    history_lines = []
    for t in range(turn_idx):
        qid = sample["turn_ids"][t]
        q = sample["questions"][t]
        a = sample["texts"][t]
        history_lines.append(f"Turn {qid} – Q: {q}\nTurn {qid} – A: {a}")
    history_block = "\n".join(history_lines) if history_lines else "(No previous turns)"

    # 3. Current question
    qid_now = sample["turn_ids"][turn_idx]
    question = sample["questions"][turn_idx]

    # 4. Build prompt
    prompt = f"""You are an expert question-answering assistant.  
You will be given:
• A Wikipedia page title: {page_title}  
• A brief background paragraph: {background}  
• The name of the section the passage is drawn from: {section}  
• The passage itself:  

{context}

**Dialogue so far**  
{history_block}

**Current question (Turn {qid_now})**  
{question}

Guidelines  
1. If the answer is explicitly stated in the passage, quote the minimal span that answers the question.  
2. If the question can be answered with “yes” or “no”, output exactly `yes` or `no`, followed by a short evidence phrase from the passage (e.g., `yes – he officially registered the party on 30 April 2012`).  
3. If the answer is not available in the passage, output exactly `CANNOTANSWER`.  
4. Keep the answer on a single line; do not add any extra explanation beyond the rules above.
"""
    # 5. Optional chain-of-thought trigger
    if thought:
        prompt += "\nLet's think step by step.\n"

    prompt += "\n**Answer:**"
    return prompt


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