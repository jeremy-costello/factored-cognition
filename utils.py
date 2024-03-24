from typing import TypeVar, List


# not really sure what this does
T = TypeVar("T", bound="QuestionAnswerNode")


class QuestionAnswerNode:
    """Tree node for recursive amplification.
    """
    def __init__(self, question: str):
        """Class initialization function.

        Args:
            question (str): Question asked by the node.
        """
        self.question = question
        self.answer = None
        self.children = []
        self.upstream_questions = []
    
    def add_child(self, child: T) -> None:
        """Add a child node to this node.

        Args:
            child (QuestionAnswerNode): Child node.
        """
        self.children.append(child)
    
    def answer_question(self, answer: str) -> None:
        """Set an answer to the node's question.

        Args:
            answer (str): Answer to the node's question.
        """
        self.answer = answer
    
    def set_upstream_questions(self, parent_node: T) -> None:
        """Set the questions upstream from this node. Questions closer to the start of the list are higher up in the tree.

        Args:
            parent_node (QuestionAnswerNode): This node's parent node.
        """
        self.upstream_questions = parent_node.upstream_questions + [parent_node.question]


def render_steps(step_list: List[str]) -> str:
    step_list_string = ""
    for index, step in enumerate(step_list, start=1):
        step_list_string += f"{index}. {step}\n"
    
    return step_list_string
