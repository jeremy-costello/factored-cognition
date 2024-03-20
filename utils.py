from typing import TypeVar


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
        