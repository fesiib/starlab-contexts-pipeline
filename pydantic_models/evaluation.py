from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Union

class MetricScale(str, Enum):
    LIKERT_7 = "likert-7"
    LIKERT_5 = "likert-5"
    LIKERT_3 = "likert-3"
    BINARY = "binary"
    COMPARISON = "comparison"

    

class BaseEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reasoning: str = Field(..., title="Brief reasoning for the evaluation.")
    confidence: int = Field(..., ge=1, le=10, title="Confidence level between 1 and 10")

class BinaryEvaluationResponse(BaseEvaluationResponse):
    decision: Literal["yes", "no"] = Field(..., title="The binary decision")

class Likert7EvaluationResponse(BaseEvaluationResponse):
    rating: int = Field(..., ge=1, le=7, title="Rating on the 7-point scale")


class Likert5EvaluationResponse(BaseEvaluationResponse):
    rating: int = Field(..., ge=1, le=5, title="Rating on the 5-point scale")

class Likert3EvaluationResponse(BaseEvaluationResponse):
    rating: int = Field(..., ge=1, le=3, title="Rating on the 3-point scale")

class ComparisonEvaluationResponse(BaseEvaluationResponse):
    decision: Literal["A", "B", "tie"] = Field(..., title="The better option between A and B or tie if they are equally good.")