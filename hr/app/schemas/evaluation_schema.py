from typing import Optional
from pydantic import BaseModel, Field, field_validator


class EvaluationSchema(BaseModel):
    score: int = Field(
         ge=0, le=10, description="Overall score from 0 to 10"
    )
    feedback: str = Field(
        description="Short structured feedback"
    )
    status: Optional[str] = Field(
        default=None, description="Error status if applicable"
    )
    message: Optional[str] = Field(
        default=None, description="Error message if applicable"
    )

    @field_validator("score", mode="before")
    @classmethod
    def reject_none_score(cls, v):
        if v is None:
            return 0
        return v