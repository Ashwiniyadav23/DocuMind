from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=12000)
    citations: list[str] = Field(default_factory=list, max_length=20)
