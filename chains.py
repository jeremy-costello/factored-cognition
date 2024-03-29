from typing import List, Union, Dict, Any

from models import Model
from recipes import QAVariableContext, RawGeneration, ParagraphAnswersQuestion, GenerateSubquestions
from papers import extract_paper_from_pdf, transform_paper_dict_into_paragraph_list
from utils import QuestionAnswerNode


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
        self.response_length = "no more than 1-2 sentences"
        self.system_message = \
            "You are a debator who wants to come to an agreeable solution to a debate with an opposing debator. You may disagree " \
            "with your opponent if that is the best conclusion to the debate. You will be given a prompt for a debate and previous " \
            "rounds of the debate (if applicable). You {agree_type} with, or are \"{position}\" this debate prompt. Please have a " \
            "truthful and good faith debate with your opposing debator. The debate will last {num_rounds} rounds. Do not repeat " \
            "yourself. {system_message_append}"
        
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
                    f"This is the first round. Please provide an opening statement for your position after being provided with the debate prompt. " \
                    f"Ensure your statement is {self.response_length} in total."
                against_system_message_append = \
                    f"This is the first round. Please provide a response of {self.response_length} to your opponent's opening statement."
            elif round == self.num_rounds - 1:
                for_system_message_append = \
                    f"This is the final round. Please provide a final response of {self.response_length} to your opponent's previous statement."
                against_system_message_append = \
                    f"This is the final round. Please provide a final response to your opponent's previous statement, along with a closing statement. " \
                    f"Ensure your response is {self.response_length} in total."
            else:
                for_system_message_append = \
                    f"There are {self.num_rounds - round} rounds remaining. Please provide a response of {self.response_length} to your opponent's previous statement."
                against_system_message_append = \
                    f"There are {self.num_rounds - round} rounds remaining. Please provide a response of {self.response_length} to your opponent's previous statement."
        
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


class AnswerQuestionFromPaper(Chain):
    """Chain for answering a question based on a paper.

    Args:
        Chain (class): Base chain class.
    """
    def __init__(self, model: Model):
        """Class initialization function.

        Args:
            model (Model): Text generation model.
        """
        super().__init__(model=model)
        
        self.context_intro = "You will be given {number} paragraphs as context.\n\n"
        self.context_format = "Paragraph {number}: {paragraph}\n\n"
        
        self.output_answer = "The model's answer is:\n{generation}\n\nThis is based on the following context(s):\n\n"
        self.output_contexts = "{number}. On page {page}, section {section_num} ({section_name}), paragraph {paragraph_num}:\n{paragraph}\n\n"
    
    def run_chain(self, paper: str, question: str, num_paragraphs: int) -> str:
        """Run the chain.

        Args:
            paper (str): Path to the paper PDF in string format.
            question (str): Question to answer.
            num_paragraphs (int): Number of paragraphs to use as context.

        Returns:
            str: Output with model answer and paragraphs used as context.
        """
        assert num_paragraphs >= 0 and isinstance(num_paragraphs, int)
        
        paper_dict = extract_paper_from_pdf(
            pdf_path=paper,
            use_llm=False
        )
        
        paragraph_list = transform_paper_dict_into_paragraph_list(
            paper_dict=paper_dict
        )
        
        true_paragraph_list = [paragraph[4] for paragraph in paragraph_list]
        
        recipe = ParagraphAnswersQuestion()
        
        _, probabilities = recipe.call_recipe(
            prompts=true_paragraph_list,
            question=question,
            model=self.model
        )
        
        numbered_probabilities = [(index, probability) for index, probability in enumerate(probabilities)]
        sorted_probabilities = sorted(numbered_probabilities, key=lambda x: x[1], reverse=True)
        
        recipe = QAVariableContext(
            context=True
        )
                
        input_without_context = self.model.meta_prompt_template.format(
            system_message=recipe.system_message,
            prompt=recipe.prompt_template.format(
                context="",
                prompt=question
            )
        )
        
        iwc_tokens = self.model.tokenize(input_without_context, add_special_tokens=True)
        remaining_input_length = self.model.context_length - len(iwc_tokens)
        
        if num_paragraphs == 0:
            num_paragraphs = len(sorted_probabilities)
        
        top_indices = [index for index, _ in sorted_probabilities[:num_paragraphs]]
        top_paragraphs = [true_paragraph_list[index] for index in top_indices]
        
        context = self.context_intro.format(
            number=num_paragraphs
        )
        
        number_of_paragraphs = num_paragraphs
        for index, paragraph in enumerate(top_paragraphs, start=1):
            potential_added_context = self.context_format.format(
                number=index,
                paragraph=paragraph
            )
            
            pac_tokens = self.model.tokenize(potential_added_context, add_special_tokens=False)
            remaining_input_length -= len(pac_tokens)
            # last two new-line characters will be stripped
            if remaining_input_length >= -2:
                context += potential_added_context
            else:
                number_of_paragraphs = index
                break
            
        _, generations = recipe.call_recipe(
            prompts=[question],
            model=self.model,
            contexts=[context.rstrip()]
        )
        
        generation = generations[0]
        output = self.output_answer.format(
            generation=generation
        )
        
        top_full_paragraph_infos = [paragraph_list[index] for index in top_indices[:number_of_paragraphs]]
        
        for index, (page, section_name, section_num, paragraph_num, paragraph) in enumerate(top_full_paragraph_infos, start=1):
            output += self.output_contexts.format(
                number=index,
                page=page,
                section_num=section_num,
                section_name=section_name,
                paragraph_num=paragraph_num,
                paragraph=paragraph
            )
        
        return output.strip()


# probably some way to batch model calls for vLLM. not sure how right now.
class RecursiveSubQuestionAnswering(Chain):
    """Chain for recusive amplification: recursively generating and answering sub-questions to help answer a question.

    Args:
        Chain (class): Base chain class.
    """
    def __init__(self, model: Model):
        """Class initialization function.

        Args:
            model (Model): Text generation model.
        """
        super().__init__(model=model)
        
        self.system_message = \
            "You are a truthful and helpful oracle. You will be provided with a set of matching numbered sub-questions and sub-answers " \
            "as context for answering a final question. Please answer the final question truthfully and succinctly, while using the " \
            "context provided by the sub-questions and sub-answers."
        self.prompt_template = "{context}\n\nFinal Question: {prompt}"
        
        self.context_intro = "You will be given the answer to {number} sub-questions as context.\n\n"
        self.context_format = "Sub-question {number}: {sub_question}\nSub-answer {number}: {sub_answer}\n\n"
    
    def create_tree(self, root: QuestionAnswerNode, depth: int, max_depth: int) -> None:
        """Recursively create a tree up to a maximum depth.

        Args:
            root (QuestionAnswerNode): Current node in the tree.
            depth (int): Current depth of the tree.
            max_depth (int): Maximum depth of the tree.
        """
        if depth < max_depth:
            recipe = GenerateSubquestions()
            
            _, generations = recipe.call_recipe(
                prompts=[root.question],
                model=self.model
            )
            
            generation = generations[0]
            sub_questions = generation.lstrip("Sub-questions:").strip()
            
            sub_question_list = [" ".join(sub_question.split(" ")[1:]) for sub_question in sub_questions.split("\n")]
            for sub_question in sub_question_list:
                child_node = QuestionAnswerNode(sub_question)
                child_node.set_upstream_questions(root)
                root.add_child(child_node)
                self.create_tree(child_node, depth + 1, max_depth)
    
    def reverse_inorder_traversal(self, node: QuestionAnswerNode) -> None:
        """Reverse in-order traversal of the tree. For answering questions with the context of answered sub-questions.

        Args:
            node (QuestionAnswerNode): Current node in the tree.
        """
        for child in reversed(node.children):
            self.reverse_inorder_traversal(child)
        if node.children:
            recipe = QAVariableContext(
                context=True,
                system_message=self.system_message,
                prompt_template=self.prompt_template,
                chain_of_thought=False
            )
            
            context = self.context_intro.format(
                number=len(node.children)
            )
            for index, child in enumerate(node.children, start=1):
                context += self.context_format.format(
                    number=index,
                    sub_question=child.question,
                    sub_answer=child.answer
                )
            _, answers = recipe.call_recipe(
                prompts=[node.question],
                model=self.model,
                contexts=[context]
            )
            node.answer_question(
                answer=answers[0]
            )
        else:
            print(node.upstream_questions)
            recipe = QAVariableContext(
                context=False
            )
            _, answers = recipe.call_recipe(
                prompts=[node.question],
                model=self.model
            )
            node.answer_question(
                answer=answers[0]
            )
                
    def run_chain(self, question: str, max_depth: int) -> QuestionAnswerNode:
        """Run the chain.

        Args:
            question (str): Question to be answered.
            max_depth (int): Maximum depth of the tree.

        Returns:
            QuestionAnswerNode: Root node of the tree.
        """
        root = QuestionAnswerNode(question)
        
        self.create_tree(
            root=root,
            depth=0,
            max_depth=max_depth
        )
        
        self.reverse_inorder_traversal(
            node=root
        )
        
        return root
