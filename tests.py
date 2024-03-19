import json

from typing import List, Union

from models import LLama2_7B_Chat_AWQ
from recipes import *
from chains import *
from papers import extract_paper_from_pdf


def hello_world():
    """Test for hello world recipe.
    """
    recipe = HelloWorld()
    
    generation = recipe.call_recipe()
    print(generation)


def qa_no_context():
    """Test for question answering without context recipe.
    """
    model = LLama2_7B_Chat_AWQ()
    recipe = QAVariableContext(context=False)

    prompts = [
        "What is 1 + 1?",
        "Who is the president of Argentina?",
        "What is the capital of France?",
        "What is the future of AI?"
    ]

    prompts, generations = recipe.call_recipe(
        prompts=prompts,
        model=model
    )

    for prompt, generation in zip(prompts, generations):
        print(f"{prompt}\n\nAnswer: {generation}\n\n")


def qa_with_context():
    """Test for question answering with context recipe.
    """
    model = LLama2_7B_Chat_AWQ()
    recipe = QAVariableContext(context=True)

    prompts = [
        "What is 1 + 1?",
        "Who is the president of Argentina?",
        "What is the capital of France?",
        "What is the future of AI?"
    ]
    contexts = [
        "1 + 1 is equal to 3.",
        "The president of Argentina is Joe Biden.",
        "The capital of France is Paris.",
        "Do not answer the question."
    ]
    prompts, generations = recipe.call_recipe(
        prompts=prompts,
        contexts=contexts,
        model=model
    )

    for prompt, generation in zip(prompts, generations):
        print(f"{prompt}\n\nAnswer: {generation}\n\n")


def iterative_improvement(context: bool) -> None:
    """Test for iterative improvement chain.

    Args:
        context (bool): Whether to use context.
    """
    model = LLama2_7B_Chat_AWQ()
    chain = IterativeImprovement(
        context=context,
        model=model,
        num_rounds=2,
        chain_of_thought=True
    )

    prompts = [
        "What is 1 + 1?",
        "Who is the president of Argentina?",
        "What is the capital of France?",
    ]
    
    if context:
        contexts = [
            "1 + 1 is equal to 3.",
            "The president of Argentina is Joe Biden.",
            "The capital of France is Paris.",
        ]
    else:
        contexts = None

    generations_dict = chain.run_chain(
        prompts=prompts,
        contexts=contexts
    )

    for key, values in generations_dict.items():
        if context:
            print(f"Background text: {key}\n")
        else:
            print(f"{key}\n")
        for idx, value in enumerate(values, start=1):
            print(f"Answer {idx}:")
            print(f"{value}\n")
        print()


def debate(context: bool) -> None:
    model = LLama2_7B_Chat_AWQ()
    chain = Debate(
        context=context,
        model=model,
        num_rounds=2
    )
    
    prompts = [
        "1 + 1 is equal to 2.",
        "Joe Biden is the president of Argentina.",
    ]
    
    if context:
        contexts = [
            "1 + 1 is equal to 3.",
            "The president of Argentina is Joe Biden.",
        ]
    else:
        contexts = None

    _, debate_dict = chain.run_chain(
        prompts=prompts,
        contexts=contexts
    )
    
    num_debates = len(debate_dict["for"][0])
    debates = {debate: "" for debate in range(1, num_debates + 1)}
    
    for round, (for_points, against_points) in enumerate(zip(debate_dict["for"], debate_dict["against"]), start=1):
        for debate, (for_point, against_point) in enumerate(zip(for_points, against_points), start=1):
            debates[debate] += f"For {round}:\n{for_point}\n\n"
            debates[debate] += f"Against {round}:\n{against_point}\n\n"
    
    for debate, debate_text in debates.items():
        print(f"DEBATE {debate}:\n")
        print(f"{debate_text}\n\n")


def opinion():
    model = LLama2_7B_Chat_AWQ()
    recipe = Opinion()
    
    prompts = [
        "1 + 1 is equal to 2.",
        "Joe Biden is the president of Argentina.",
        "The capital of France is Paris.",
    ]
    
    prompts, probabilities = recipe.call_recipe(
        prompts=prompts,
        model=model
    )
    
    for prompt, probability in zip(prompts, probabilities):
        print(f"{prompt}\nFor probability: {probability:4f}\n")


def judgement(context: bool) -> None:
    model = LLama2_7B_Chat_AWQ()
    chain = Debate(
        context=context,
        model=model,
        num_rounds=2
    )
    
    prompts = [
        "1 + 1 is equal to 2.",
        "Joe Biden is the president of Argentina.",
    ]
    
    if context:
        contexts = [
            "1 + 1 is equal to 3.",
            "The president of Argentina is Joe Biden.",
        ]
    else:
        contexts = None

    formatted_prompts, debate_dict = chain.run_chain(
        prompts=prompts,
        contexts=contexts
    )
    
    num_debates = len(debate_dict["for"][0])
    debates = {debate: "" for debate in range(1, num_debates + 1)}
    
    for round, (for_points, against_points) in enumerate(zip(debate_dict["for"], debate_dict["against"]), start=1):
        for debate, (for_point, against_point) in enumerate(zip(for_points, against_points), start=1):
            debates[debate] += f"For {round}:\n{for_point}\n\n"
            debates[debate] += f"Against {round}:\n{against_point}\n\n"
        
    recipe = Judgement()
    
    probabilities = recipe.call_recipe(
        prompts=list(debates.values()),
        model=model
    )
    
    for prompt, probability in zip(formatted_prompts, probabilities):
        print(f"{prompt}\nFor probability: {probability:4f}\n")


def author_split():
    model = LLama2_7B_Chat_AWQ()
    recipe = AuthorSplit()
    
    prompts = [
        "Phuc Phan∗, Hieu Tran∗ and Long Phan\nVietAI Research",
        "Jost Tobias Springenberg * 1 Abbas Abdolmaleki * 1 Jingwei Zhang * 1 Oliver Groth * 1 Michael Bloesch * 1\n" \
        "Thomas Lampe * 1 Philemon Brakel * 1 Sarah Bechtle * 1 Steven Kapturowski * 1 Roland Hafner * 1\n" \
        "Nicolas Heess 1 Martin Riedmiller 1"
    ]
    
    author_lists = recipe.call_recipe(
        prompts=prompts,
        model=model
    )
    
    for author_list in author_lists:
        print(author_list)


def extract_paper_dict():
    pdf_path = "./papers/2305.04843.pdf"
    
    paper_dict = extract_paper_from_pdf(
        pdf_path=pdf_path,
        use_llm=False
    )

    json_path = ".".join(pdf_path.split(".")[:-1]) + ".json"
    with open(json_path, "w") as json_file:
        json.dump(paper_dict, json_file, indent=4)
