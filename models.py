from typing import List
from vllm import LLM, SamplingParams


class Model:
    def __init__(self):
        self.llm = None
    
    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        assert isinstance(self.llm, LLM)
        return self.llm.generate(prompts, sampling_params)


class LLaMa2(Model):
    def __init__(self):
        super().__init__()
        
        self.vocab_size = 32000
        self.yes_token = 3869
        self.no_token = 1939
        
        self.prompt_template = "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST] "


class LLama2_7B_Chat_AWQ(LLaMa2):
    def __init__(self):
        super().__init__()
        
        model = "TheBloke/Llama-2-7b-Chat-AWQ"
        quantization = "awq"
        
        self.llm = LLM(model=model, quantization=quantization, enforce_eager=True)
    