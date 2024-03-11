from models import LLama2_7B_Chat_AWQ
from recipes import QAWithContext


model = LLama2_7B_Chat_AWQ()
recipe = QAWithContext(chain_of_thought=True)

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

unformatted_prompts, text_generations = recipe.call_recipe(
    prompts=prompts,
    contexts=contexts,
    model=model
)

for prompt, generation in zip(unformatted_prompts, text_generations):
    print(prompt + "\n\n" + "Answer: " + generation)
    print("\n\n")
