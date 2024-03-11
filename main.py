from models import LLama2_7B_Chat_AWQ
from recipes import Classification


model = LLama2_7B_Chat_AWQ()
recipe = Classification()

prompts = [
    "Does 1 + 1 = 2?",
    "Is Benjamin Franklin the president of Argentina?",
    "Is Paris the capital of France?",
    "What is the future of AI?"
]

outputs = recipe.call_recipe(
    prompts=prompts,
    model=model
)

print(outputs)
