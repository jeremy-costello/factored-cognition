import numpy as np
from typing import List, Tuple, Union
from vllm import SamplingParams

from models import Model


class Recipe:
    """Base recipe class.
    """
    def __init__(self):
        self.system_message = None
        self.temperature = 0.8
        self.top_p = 0.95
        self.repetition_penalty = 1.1
    
    def get_generation_inputs(self, prompts: List[str], meta_prompt_template: Union[str, None] = None, *args, **kwargs) -> Tuple[List[str], SamplingParams]:
        """Get inputs for generation.

        Args:
            prompts (List[str]): List of unformatted prompts.
            meta_prompt_template (Union[str, None]): Prompt formatting template. Defaults to None.

        Returns:
            Tuple[List[str], SamplingParams]: List of formatted prompts, sampling parameters.
        """
        
        if meta_prompt_template is not None:
            assert isinstance(self.system_message, str)
            assert isinstance(meta_prompt_template, str)
            prompts = [
                meta_prompt_template.format(
                    system_message=self.system_message,
                    prompt=prompt
                )
                for prompt in prompts
            ]
        
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            *args,
            **kwargs
        )
        return prompts, sampling_params
        
    def call_recipe(self) -> None:
        """Empty recipe call.

        Raises:
            NotImplementedError: This function is not implemented in this class.
        """
        raise NotImplementedError()


class HelloWorld(Recipe):
    """Hello world recipe class.

    Args:
        Recipe (class): Base recipe class.
    """
    def __init__(self):
        super().__init__()
    
    def call_recipe(self) -> str:
        """Hello world recipe call.

        Returns:
            str: Hello world.
        """
        return "Hello, world!"


class RawGeneration(Recipe):
    """Raw generation recipe class.

    Args:
        Recipe (class): Base recipe class.
    """
    def __init__(self):
        super().__init__()
    
    def call_recipe(self, prompts: List[str], model: Model) -> Tuple[List[str], List[str]]:
        """Raw generation recipe call.

        Args:
            prompts (List[str]): List of prompts.
            model (Model): Text generation model.

        Returns:
            Tuple[List[str], List[str]]: Prompts and list of text strings generated by the model.
        """
        prompts, sampling_params = self.get_generation_inputs(
            prompts=prompts,
            max_tokens=model.context_length
        )
        
        outputs = model.generate(prompts, sampling_params)
        text_generations = [output.outputs[0].text.strip() for output in outputs]
        
        return prompts, text_generations


class QAVariableContext(Recipe):
    """Question answering with or without context recipe class.

    Args:
        Recipe (class): Base recipe class.
    """
    def __init__(self, context: bool, system_message: Union[str, None] = None,
                 prompt_template: Union[str, None] = None, chain_of_thought: bool = False):
        """Class initialization function.

        Args:
            context (bool): Whether to use context.
            system_message (Union[str, None], optional): Custom system message. Defaults to None.
            prompt_template (Union[str, None], optional): Custom prompt template. Defaults to None.
            chain_of_thought (bool, optional): Whether to use chain-of-thought prompting. Defaults to False.
        """
        super().__init__()
        
        if context:
            if system_message is None:
                self.system_message = \
                    "You are a truthful and helpful oracle. You will be provided with a background text passage as context. " \
                    "Please answer the question following the background text passage truthfully and succinctly."
            else:
                self.system_message = system_message
            
            if prompt_template is None:
                self.prompt_template = "Background text: {context}\n\nQuestion: {prompt}"
            else:
                self.prompt_template = prompt_template
        else:
            if system_message is None:
                self.system_message = "You are a truthful and helpful oracle. Please answer the following question truthfully and succinctly."
            else:
                self.system_message = system_message
            
            if prompt_template is None:
                self.prompt_template = "Question: {prompt}"
            else:
                self.prompt_template = prompt_template
        
        self.chain_of_thought = chain_of_thought
        if self.chain_of_thought:
            self.chain_of_thought_prefix = "Let's think step by step.\n"
            self.system_message += f" Your answer will begin with \"{self.chain_of_thought_prefix}\" Please number the steps of your thought process."
            
    def call_recipe(self, prompts: List[str], model: Model, contexts: Union[str, List[str], None] = None) -> Tuple[List[str], List[str]]:
        """Question answering with or without context recipe call.

        Args:
            prompts (List[str]): List of unformatted prompts.
            model (Model): Text generation model.
            contexts (Union[str, List[str], None], optional): Context string, list of context strings, or None. Defaults to None.

        Returns:
            Tuple[List[str], List[str]]: Unformatted prompts and list of text strings generated by the model.
        """
        if isinstance(contexts, str):
            prompts = [
                self.prompt_template.format(
                    context=contexts,
                    prompt=prompt
                )
                for prompt in prompts
            ]
        elif isinstance(contexts, list):
            assert len(prompts) == len(contexts)
            prompts = [
                self.prompt_template.format(
                    context=context,
                    prompt=prompt
                )
                for prompt, context in zip(prompts, contexts)
            ]
        elif contexts is None:
            prompts = [
                self.prompt_template.format(
                    prompt=prompt
                )
                for prompt in prompts
            ]
        
        unformatted_prompts = prompts
        meta_prompt_template = model.meta_prompt_template
        if self.chain_of_thought:
            meta_prompt_template += self.chain_of_thought_prefix
        
        prompts, sampling_params = self.get_generation_inputs(
            prompts=prompts,
            meta_prompt_template=meta_prompt_template,
            max_tokens=model.context_length
        )
        
        outputs = model.generate(prompts, sampling_params)
        text_generations = [output.outputs[0].text.strip() for output in outputs]
        
        if self.chain_of_thought and not self.raw_generation_prompt:
            text_generations = [self.chain_of_thought_prefix + text_generation for text_generation in text_generations]
        
        return unformatted_prompts, text_generations


# DEPRECATED
class QANoContext(Recipe):
    """Question answering without context recipe class.

    Args:
        Recipe (class): Base recipe class.
    """
    def __init__(self):
        super().__init__()
        self.system_message = "You are a truthful and helpful oracle. Please answer the following question truthfully and succinctly."
        
        raise DeprecationWarning("This method has been deprecated. Please use QAVariableContext with context=False instead.")
    
    def call_recipe(self, prompts: List[str], model: Model) -> List[str]:
        """Question answering without context recipe call.

        Args:
            prompts (List[str]): List of unformatted prompts.
            model (Model): Text generation model.

        Returns:
            List[str]: List of text strings generated by the model.
        """
        original_prompts = prompts
        
        prompts, sampling_params = self.get_generation_inputs(
            prompts=prompts,
            meta_prompt_template=model.meta_prompt_template,
            max_tokens=model.context_length
        )
        outputs = model.generate(prompts, sampling_params)
        text_generations = [output.outputs[0].text.strip() for output in outputs]
        return original_prompts, text_generations


# DEPRECATED
class QAWithContext(Recipe):
    """Question answering with context recipe class.

    Args:
        Recipe (class): Base recipe class.
    """
    def __init__(self,
                 system_message: Union[str, None] = None,
                 context_template: Union[str, None] = None,
                 chain_of_thought: bool = False):
        """Class initialization function.

        Args:
            system_message (Union[str, None], optional): Custom system message. Defaults to None.
            context_template (Union[str, None], optional): Custom context template. Defaults to None.
            chain_of_thought (bool, optional): Whether to use chain-of-thought prompting. Defaults to False.

        Raises:
            DeprecationWarning: This method has been deprecated.
        """
        super().__init__()
        self.chain_of_thought = chain_of_thought
        
        # custom system message for improvement chain
        if system_message is None:
            self.system_message = \
                "You are a truthful and helpful oracle. You will be provided with a background text passage as context. " \
                "Please answer the question following the background text passage truthfully and succinctly."
        else:
            self.system_message = system_message
        
        if self.chain_of_thought:
            self.chain_of_thought_prefix = "Let's think step by step.\n"
            self.system_message += f" Your answer will begin with \"{self.chain_of_thought_prefix}\" Please number the steps of your thought process."
        
        # custom context template for improvement chain
        if context_template is None:
            self.context_template = "Background text: {context}\n\nQuestion: {prompt}"
        else:
            self.context_template = context_template
        
        raise DeprecationWarning("This method has been deprecated. Please use QAVariableContext with context=True instead.")
    
    def call_recipe(self, prompts: List[str], contexts: Union[str, List[str]], model: Model) -> Tuple[List[str], List[str]]:
        """Question answering with context recipe call.

        Args:
            prompts (List[str]): List of unformatted prompts.
            contexts (Union[str, List[str]]): Context string or list of context strings.
            model (Model): Text generation model.

        Returns:
            List[str]: List of text strings generated by the model.
        """
        if isinstance(contexts, str):
            prompts = [
                self.context_template.format(
                    context=contexts,
                    prompt=prompt
                )
                for prompt in prompts
            ]
        elif isinstance(contexts, list):
            assert len(prompts) == len(contexts)
            prompts = [
                self.context_template.format(
                    context=context,
                    prompt=prompt
                )
                for prompt, context in zip(prompts, contexts)
            ]
        unformatted_prompts = prompts
        meta_prompt_template = model.meta_prompt_template
        if self.chain_of_thought:
            meta_prompt_template += self.chain_of_thought_prefix
        prompts, sampling_params = self.get_generation_inputs(
            prompts=prompts,
            meta_prompt_template=meta_prompt_template,
            max_tokens=model.context_length
        )
        outputs = model.generate(prompts, sampling_params)
        text_generations = [output.outputs[0].text.strip() for output in outputs]
        if self.chain_of_thought:
            text_generations = [self.chain_of_thought_prefix + text_generation for text_generation in text_generations]
        return unformatted_prompts, text_generations


class Opinion(Recipe):
    def __init__(self):
        self.system_message = \
            "You are a truthful and unbiased judge. You will be shown a statement. Please reply For if you agree with the statement, " \
            "or Against if you disagree with the statment. Please only answer For or Against."
        self.temperature = 1.0
        self.top_p = 1.0
        self.repetition_penalty = 1.0
        
    def call_recipe(self, prompts: List[str], model: Model) -> Union[List[str], List[float]]:
        original_prompts = prompts
        
        prompts, sampling_params = self.get_generation_inputs(
            prompts=prompts,
            meta_prompt_template=model.meta_prompt_template,
            max_tokens=1,
            logprobs=1
        )
        outputs = model.generate(prompts, sampling_params)
        
        for_probabilities = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            if generated_text == "For"[:len(generated_text)]:
                prob = np.exp(list(output.outputs[0].logprobs[0].values())[0])
            elif generated_text == "Against"[:len(generated_text)]:
                prob = 1.0 - np.exp(list(output.outputs[0].logprobs[0].values())[0])
            else:
                prob = -1.0
            for_probabilities.append(prob)
        
        return original_prompts, for_probabilities


class Judgement(Recipe):
    def __init__(self):
        self.system_message = \
            "You are a truthful and unbiased judge. You will be shown a statement and a debate about this statement between " \
            "two debators: one \"For\" the statement and one \"Against\" the statement. Please reply For if you agree with the " \
            "statement after seeing the debate, or Against if you disagree with the statement after seeing the debate. Please " \
            "only answer For or Against."
        self.temperature = 1.0
        self.top_p = 1.0
        self.repetition_penalty = 1.0
        
        self.prompt_append = "The debate is now over. Please remember to only answer For or Against."
    
    def call_recipe(self, prompts: List[str], model: Model) -> List[float]:
        appended_prompts = [f"{prompt}\n\n{self.prompt_append}" for prompt in prompts]
        
        prompts, sampling_params = self.get_generation_inputs(
            prompts=appended_prompts,
            meta_prompt_template=model.meta_prompt_template,
            max_tokens=1,
            logprobs=1
        )
        
        outputs = model.generate(prompts, sampling_params)
        
        for_probabilities = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            if generated_text == "For"[:len(generated_text)]:
                prob = np.exp(list(output.outputs[0].logprobs[0].values())[0])
            elif generated_text == "Against"[:len(generated_text)]:
                prob = 1.0 - np.exp(list(output.outputs[0].logprobs[0].values())[0])
            else:
                prob = -1.0
            for_probabilities.append(prob)
        
        return for_probabilities
        

class Classification(Recipe):
    """Classification recipe class.

    Args:
        Recipe (class): Base recipe class.
    """
    def __init__(self):
        self.system_message = "You are a truthful and helpful oracle. Please only answer Yes or No to the following question."
        self.temperature = 1.0
        self.top_p = 1.0
        self.repetition_penalty = 1.0
    
    def call_recipe(self, prompts: List[str], model: Model) -> List[float]:
        """Classification recipe call.

        Args:
            prompts (List[str]): List of unformatted prompts.
            model (Model): Text generation model.

        Returns:
            List[float]: List of probabilities of 'Yes'.
        """
        prompts, sampling_params = self.get_generation_inputs(
            prompts=prompts,
            meta_prompt_template=model.meta_prompt_template,
            max_tokens=1,
            logprobs=1
        )
        outputs = model.generate(prompts, sampling_params)
        
        yes_probabilities = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            if generated_text == "Yes":
                prob = np.exp(list(output.outputs[0].logprobs[0].values())[0])
            elif generated_text == "No":
                prob = 1.0 - np.exp(list(output.outputs[0].logprobs[0].values())[0])
            else:
                prob = -1.0
            yes_probabilities.append(prob)
        
        return yes_probabilities
    