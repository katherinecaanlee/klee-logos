from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateLogosRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Brand transcript or creative brief text.")
    brand_name: Optional[str] = Field("", description="Optional brand name to reinforce in prompts.")
    allow_word_mark: Optional[bool] = Field(
        default=None,
        description="Override default word-mark allowance; if None, uses server default.",
    )


class LogoResult(BaseModel):
    prompt: str
    justification: str
    image_path: str
    image_url: str
    image_base64: str


class GenerateLogosResponse(BaseModel):
    logos: List[LogoResult]
