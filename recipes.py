import numpy as np
from typing import List
from vllm import SamplingParams

from models import Model


class Recipe:
    def __init__(self):
        self.system_message = None
        self.sampling_params = None
    
    def call_recipe(self) -> None:
        raise NotImplementedError()


class Classification(Recipe):
    def __init__(self):
        self.system_message = "You are a truthful and helpful oracle. Please only answer Yes or No to the following questions."
        self.sampling_params = SamplingParams(temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=1, logprobs=1)
    
    def call_recipe(self, prompts: List[str], model: Model) -> List[float]:
        prompts = [
            model.prompt_template.format(
                system_message=self.system_message,
                prompt=prompt
            )
            for prompt in prompts
        ]
        outputs = model.generate(prompts, self.sampling_params)
        
        probs = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            if generated_text == "Yes":
                prob = np.exp(list(output.outputs[0].logprobs[0].values())[0])
            elif generated_text == "No":
                prob = 1.0 - np.exp(list(output.outputs[0].logprobs[0].values())[0])
            else:
                prob = -1.0
            probs.append(prob)
        
        return probs
    