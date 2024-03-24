from typing import List
from vllm import LLM, SamplingParams


class Model:
    """Base model class.
    """
    def __init__(self):
        pass
    
    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        """Generate responses from prompts.

        Args:
            prompts (List[str]): List of prompts.
            sampling_params (SamplingParams): Sampling parameters.

        Returns:
            List[str]: List of model responses.
        """
        assert isinstance(self.llm, LLM)
        return self.llm.generate(prompts, sampling_params)

    def tokenize(self, input_string: str, add_special_tokens: bool) -> List[int]:
        tokenizer = self.llm.llm_engine.tokenizer.tokenizer
        tokens = tokenizer.encode(input_string, add_special_tokens=add_special_tokens)
        return tokens


class LLaMa2(Model):
    """LLaMa2 model class.

    Args:
        Model (_type_): Base model class.
    """
    def __init__(self):
        super().__init__()
        
        self.vocab_size = 32000
        self.context_length = 4096
        
        self.meta_prompt_template = "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST] "
        # prompt append for continued conversation (e.g. debate)
        self.meta_continuation_template = "{response}</s><s>[INST] {prompt} [/INST] "


class LLama2_7B_Chat_AWQ(LLaMa2):
    """LLaMa2-7B-Chat with 4-bit AWQ quantization.

    Args:
        LLaMa2 (_type_): LLama2 model class.
    """
    def __init__(self):
        super().__init__()
        
        model = "TheBloke/Llama-2-7b-Chat-AWQ"
        quantization = "awq"
        
        self.llm = LLM(model=model, quantization=quantization, enforce_eager=True)
    