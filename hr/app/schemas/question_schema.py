from pydantic import BaseModel, Field

class QuestionSchema(BaseModel):
    question: str = Field(description="Generated interview question")