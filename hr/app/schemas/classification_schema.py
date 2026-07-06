from typing import Literal
from pydantic import BaseModel, Field

class ClassificationSchema(BaseModel):
    category: Literal["technical", "problem_solving", "behavioral"] = Field(
        description="Interview question category"
    )