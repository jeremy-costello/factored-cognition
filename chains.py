from typing import List, Union, Dict

from models import Model
from recipes import QAVariableContext, RawGeneration


class Chain:
    """Base chain class.
    """
    def __init__(self, model: Model):
        """Class initialization function.

        Args:
            model (Model): Text generation model.
        """
        self.model = model
        
        self.system_message = None
        self.previous_links = None
    
    def run_chain(self) -> None:
        """Empty chain runner.

        Raises:
            NotImplementedError: This function is not implemented in this class.
        """
        raise NotImplementedError()


class IterativeImprovement(Chain):
    """Iterative improvement chain class.

    Args:
        Chain (class): Base chain class.
    """
    def __init__(self, context: bool, model: Model, num_rounds: int, chain_of_thought: bool = False):
        """Class initialization function.

        Args:
            context (bool): Whether to use context.
            model (Model): Text generation model.
            num_rounds (int): Number of iterative improvement rounds.
            chain_of_thought (bool, optional): Whether to use chain-of-thought prompting. Defaults to False.
        """
        super().__init__(model=model)
        
        assert num_rounds > 0 and isinstance(num_rounds, int)
        
        self.num_rounds = num_rounds
        self.chain_of_thought = chain_of_thought
        self.system_message = \
            "You are a truthful and helpful oracle. You may be provided with a background text passage as context, " \
            "a question, and previous answers to this question. Please check if the most recent answer is correct. If " \
            "the most recent answer is correct, repeat the most recent answer and add \"QED.\" at the end. If the most " \
            "recent answer is incorrect, please correct the most recent answer and explain where the most recent answer went wrong."
        
        self.context = context
        if self.context:
            self.prompt_template = "{context}\n\nQuestion: {prompt}"
        else:
            self.prompt_template = "Question: {prompt}"
    
    def run_chain(self, prompts: List[str], contexts: Union[str, List[str], None] = None) -> Dict[str, List[str]]:
        """Iterative improvement chain runner.

        Args:
            prompts (List[str]): List of unformatted prompts.
            contexts (Union[str, List[str], None], optional): Context string or list of context strings. Defaults to None.

        Raises:
            ValueError: Chain was initialized to use context but no context was provided when running the chain.
            ValueError: Chain was initialized to not use context but context was provided when running the chain.

        Returns:
            Dict[str, List[str]]: Dictionary. Keys are prompts. Values are lists of text strings generated by the model.
        """
        if self.context and contexts is None:
            raise ValueError("Chain was initialized to use context but no context was provided when running the chain.")
        elif not self.context and contexts is not None:
            raise ValueError("Chain was initialized to not use context but context was provided when running the chain.")
        
        if self.context:
            original_prompts = [
                self.prompt_template.format(
                    context=context,
                    prompt=prompt
                )
                for context, prompt in zip(contexts, prompts)
            ]
        else:
            original_prompts = [
                self.prompt_template.format(
                    prompt=prompt
                )
                for prompt in prompts
            ]
        
        generations_dict = dict()
        for original_prompt in original_prompts:
            generations_dict[original_prompt] = []
        
        for round in range(self.num_rounds):
            if round == 0:
                if self.context:
                    self.previous_links = contexts
                else:
                    self.previous_links = None
                
                system_message = None
                prompt_template = None
                context = self.context
            else:
                self.previous_links = \
                    [f"{prompt}\n\nAnswer: {generation}" for prompt, generation in zip(prompts, generations)]
                system_message = self.system_message
                prompt_template = self.prompt_template
                context = True
            
            recipe = QAVariableContext(
                context=context,
                system_message=system_message,
                prompt_template=prompt_template,
                chain_of_thought=self.chain_of_thought
            )
            
            prompts, generations = recipe.call_recipe(
                prompts=original_prompts,
                contexts=self.previous_links,
                model=self.model
            )
            for idx, original_prompt in enumerate(original_prompts):
                generations_dict[original_prompt].append(generations[idx])
        
        return generations_dict


class Debate(Chain):
    """Debate chain class.

    Args:
        Chain (class): Base chain class.
    """
    def __init__(self, context: bool, model: Model, num_rounds: int):
        """Class initialization function.

        Args:
            context (bool): Whether to use context.
            model (Model): Text generation model.
            num_rounds (int): Number of iterative improvement rounds.
        """
        super().__init__(model=model)
        
        assert num_rounds > 0 and isinstance(num_rounds, int)
        
        self.num_rounds = num_rounds
        self.system_message = \
            "You are a debator who wants to come to an agreeable solution to a debate with an opposing debator. You may disagree " \
            "with your opponent if that is the best conclusion to the debate. You will be given a prompt for a debate and previous " \
            "rounds of the debate (if applicable). You {agree_type} with, or are \"{position}\" this debate prompt. Please have a " \
            "truthful and good faith debate with your opposing debator. The debate will last {num_rounds} rounds. Do not repeat " \
            "yourself. Try to use no more than 1-2 sentences per round. {system_message_append}"
        
        self.context = context
        if self.context:
            self.prompt_template = "Background information for the debate: {context}\n\nDebate topic: {prompt}"
        else:
            self.prompt_template = "Debate topic: {prompt}"
    
    def run_chain(self, prompts: List[str], contexts: Union[str, List[str], None] = None) -> Union[List[str], Dict[str, List[str]]]:
        """Debate chain runner.

        Args:
            prompts (List[str]): List of unformatted prompts.
            contexts (Union[str, List[str], None], optional): Context string or list of context strings. Defaults to None.

        Raises:
            ValueError: Chain was initialized to use context but no context was provided when running the chain.
            ValueError: Chain was initialized to not use context but context was provided when running the chain.

        Returns:
            Union[List[str], Dict[str, List[str]]]: Tuple of (list of formatted prompts, dictionary of "for" and "against" debate rounds).
        """
        if self.context and contexts is None:
            raise ValueError("Recipe was initialized to use context but no context was provided in the recipe call.")
        elif not self.context and contexts is not None:
            raise ValueError("Recipe was initialized to not use context but context was provided in the recipe call.")
        
        if self.context:
            original_prompts = [
                self.prompt_template.format(
                    context=context,
                    prompt=prompt
                )
                for context, prompt in zip(contexts, prompts)
            ]
        else:
            original_prompts = [
                self.prompt_template.format(
                    prompt=prompt
                )
                for prompt in prompts
            ]
        
        debate_dict = {
            "for": [],
            "against": []
        }
        
        for round in range(self.num_rounds):
            if round == 0:
                for_system_message_append = \
                    "This is the first round. Please provide an opening statement for your position after being provided with the debate prompt."
                against_system_message_append = \
                    "This is the first round. Please provide a response to your opponent's opening statement."
            elif round == self.num_rounds - 1:
                for_system_message_append = \
                    "This is the final round. Please provide a final response to your opponent's previous statement."
                against_system_message_append = \
                    "This is the final round. Please provide a final response to your opponent's previous statement, along with a closing statement."
            else:
                for_system_message_append = \
                    f"There are {self.num_rounds - round} rounds remaining. Please provide a response to your opponent's previous statement."
                against_system_message_append = \
                    f"There are {self.num_rounds - round} rounds remaining. Please provide a response to your opponent's previous statement."
        
            for_system_message = self.system_message.format(
                agree_type="agree",
                position="For",
                num_rounds=self.num_rounds,
                system_message_append=for_system_message_append
            )
            against_system_message = self.system_message.format(
                agree_type="disagree",
                position="Against",
                num_rounds=self.num_rounds,
                system_message_append=against_system_message_append
            )
            
            # FOR
            if round == 0:
                for_prompts = [
                    self.model.meta_prompt_template.format(
                        system_message=for_system_message,
                        prompt=original_prompt
                    )
                    for original_prompt in original_prompts
                ]
            elif round > 0:
                # add (for, against) to for
                for_prompts = [
                    for_prompt + \
                        self.model.meta_continuation_template.format(
                            response=response,
                            prompt=prompt
                        )
                    for for_prompt, response, prompt in \
                        zip(for_prompts, debate_dict["for"][-1], debate_dict["against"][-1])
                ]
            
            for_recipe = RawGeneration()
            _, for_generations = for_recipe.call_recipe(
                prompts=for_prompts,
                model=self.model
            )
            debate_dict["for"].append(for_generations)
            
            # AGAINST
            if round == 0:
                # add (for,) to against
                against_prompts = [
                    self.model.meta_prompt_template.format(
                        system_message=against_system_message,
                        prompt=original_prompt + for_generation
                    )
                    for original_prompt, for_generation in zip(original_prompts, debate_dict["for"][0])
                ]
            elif round > 0:
                # add (against, for) to against
                against_prompts = [
                    against_prompt + \
                        self.model.meta_continuation_template.format(
                            response=response,
                            prompt=prompt
                        )
                    for against_prompt, response, prompt in \
                        zip(against_prompts, debate_dict["against"][-1], debate_dict["for"][-1])
                ]
            
            # AGAINST
            against_recipe = RawGeneration()
            _, against_generations = against_recipe.call_recipe(
                prompts=against_prompts,
                model=self.model
            )
            debate_dict["against"].append(against_generations)
    
        return original_prompts, debate_dict