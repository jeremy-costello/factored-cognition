import json

from typing import List, Union

from models import LLama2_7B_Chat_AWQ
from recipes import *
from chains import *
from papers import *


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


def get_paragraph_list():
    pdf_path = "./papers/2305.04843.pdf"
    
    paper_dict = extract_paper_from_pdf(
        pdf_path=pdf_path,
        use_llm=False
    )
    
    paragraph_list = transform_paper_dict_into_paragraph_list(
        paper_dict=paper_dict
    )
    
    with open("./papers/paragraph_list.txt", "w") as f:
        for paragraph in paragraph_list:
            f.write(f"{paragraph}\n\n")


def paragraph_answers_question():
    model = LLama2_7B_Chat_AWQ()
    recipe = ParagraphAnswersQuestion()
    
    question = "What is my favourite food?"
    
    prompts = [
        "My favourite colour is red.",
        "My favourite food is pizza.",
        "My favourite animal is the shark."
    ]
    
    prompts, probabilities = recipe.call_recipe(
        prompts=prompts,
        question=question,
        model=model
    )
    
    for prompt, probability in zip(prompts, probabilities):
        print(f"{prompt} || Probability: {probability:4f}")


def answer_question_from_paper():
    model = LLama2_7B_Chat_AWQ()
    chain = AnswerQuestionFromPaper(
        model=model
    )
    
    paper = "./papers/2305.04843.pdf"
    
    question = "Who are the authors of this paper?"
    
    output = chain.run_chain(
        paper=paper,
        question=question,
        num_paragraphs=0
    )
    print(output)


def tokenizer_test():
    model = LLama2_7B_Chat_AWQ()
    tokenizer = model.llm.llm_engine.tokenizer.tokenizer
    
    text = "The quick brown fox jumps over the lazy dog.\n\n"
    
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    print(tokenized_text)


# this always chooses the first prompt for some reason.
def paragraph_comparison():
    model = LLama2_7B_Chat_AWQ()
    recipe = ParagraphComparison()
    
    prompts = [
        ("My favourite food is pizza.", "My favourite colour is red."),
        ("My favourite colour is red.", "My favourite food is pizza."),
    ]
    
    questions = "What is my favourite food?"
    
    prompts, decisions, probabilities = recipe.call_recipe(
        prompts=prompts,
        questions=questions,
        model=model
    )
    
    for prompt, decision, probability in zip(prompts, decisions, probabilities):
        print(prompt, decision, probability)


def generate_subquestions():
    model = LLama2_7B_Chat_AWQ()
    recipe = GenerateSubquestions()
    
    prompts = [
        "What is the time difference between Toronto and Vancouver?",
        "What is the effect of creatine on cognition?",
        "Will artificial intelligence be a net benefit for society?"
    ]
    
    prompts, generations = recipe.call_recipe(
        prompts=prompts,
        model=model
    )
    
    question_list = []
    for prompt, generation in zip(prompts, generations):
        question_dict = {
            "question": prompt,
            "sub_questions": []
        }
        sub_questions = generation.lstrip("Sub-questions:").strip()
        sub_question_list = [" ".join(sub_question.split(" ")[1:]) for sub_question in sub_questions.split("\n")]
        for sub_question in sub_question_list:
            sub_question_dict = {
                "question": sub_question,
                "sub_questions": []
            }
            question_dict["sub_questions"].append(sub_question_dict)
        question_list.append(question_dict)
    
    json_path = "./papers/question_list.json"
    with open(json_path, "w") as json_file:
        json.dump(question_list, json_file, indent=4)


def recursive_sub_question_answering():
    model = LLama2_7B_Chat_AWQ()
    chain = RecursiveSubQuestionAnswering(
        model=model
    )
    
    question = "Will artificial intelligence be a net benefit for society?"
    
    output = chain.run_chain(
        question=question,
        max_depth=2
    )
    print(output.question)
    print(output.answer)
    print(len(output.children))
