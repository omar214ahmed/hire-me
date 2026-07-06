from .chains import Chains
class ClassificationQuestion:
    def __init__(self, chains:Chains):
        self.classification_chain = chains.classification_chain

    
    def classify(self, question: str):
        try:
          result = self.classification_chain.invoke({
            "question": question
            })
          return result.category
        except Exception:
              return "technical"  # fallback لو فشل