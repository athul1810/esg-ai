"""Pydantic schemas for the ESG AI backend service."""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class PortfolioBase(BaseModel):
    name: str = Field(..., example="Climate Watch")
    companies: List[str] = Field(..., example=["tesla", "bp"])
    start_date: date = Field(..., example="2025-01-01")
    end_date: date = Field(..., example="2025-03-01")
    alerts: Optional[dict] = Field(default=None)

    @validator("companies")
    def ensure_companies(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one company must be provided")
        return value


class PortfolioResponse(PortfolioBase):
    updated_at: datetime


class PortfolioListResponse(BaseModel):
    items: List[PortfolioResponse]


class ThresholdUpdate(BaseModel):
    alerts: dict


class NoteCreate(BaseModel):
    note: str
    author: Optional[str] = None


class NoteResponse(BaseModel):
    note: str
    author: Optional[str]
    created_at: datetime


class PortfolioAnalyticsRecord(BaseModel):
    company: str
    article_count: int
    avg_tone: Optional[float]
    positive_share: Optional[float]
    tone_vs_industry: Optional[float]
    risk_rating: str
    risk_headline: str
    risk_score: int


class PortfolioAlert(BaseModel):
    company: str
    rating: str
    score: int
    headline: str


class PortfolioAnalyticsResponse(BaseModel):
    records: List[PortfolioAnalyticsRecord]
    alerts: List[PortfolioAlert]


class AdvisoryRequest(BaseModel):
    company: str
    prompt: Optional[str] = None
    profile: Optional[dict] = None
    start_window: Optional[str] = Field(default="dec30")
    end_window: Optional[str] = Field(default="jan12")
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class AdvisoryResponse(BaseModel):
    executive_summary: str
    talking_points: List[str]
    risk_radar: List[str]
    recommended_actions: List[str]
    evidence: List[str]
    disclaimer: str


__all__ = [
    "PortfolioBase",
    "PortfolioResponse",
    "PortfolioListResponse",
    "ThresholdUpdate",
    "NoteCreate",
    "NoteResponse",
    "PortfolioAnalyticsResponse",
    "AdvisoryRequest",
    "AdvisoryResponse",
]
