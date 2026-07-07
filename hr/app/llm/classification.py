import logging
from .chains import Chains

logger = logging.getLogger("hr_interview.classification")

class ClassificationQuestion:
    def __init__(self, chains: Chains):
        self.classification_chain = chains.classification_chain

    def classify(self, question: str):
        try:
            result = self.classification_chain.invoke({"question": question})
            return result.category
        except Exception:
            logger.exception("classification_chain failed, defaulting to 'technical'")
            return "technical"  