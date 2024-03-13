from models import LLama2_7B_Chat_AWQ
from recipes import *
from chains import *


def hello_world():
    recipe = HelloWorld()
    
    generation = recipe.call_recipe()
    print(generation)


def qa_no_context():
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


def iterative_improvement(context):
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
        #"What is the future of AI?"
    ]
    
    if context:
        contexts = [
            "1 + 1 is equal to 3.",
            "The president of Argentina is Joe Biden.",
            "The capital of France is Paris.",
            #"Do not answer the question."
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
