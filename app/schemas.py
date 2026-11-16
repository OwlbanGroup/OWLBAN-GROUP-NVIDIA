"""Pydantic schemas for the production app layer."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Asset(BaseModel):
    symbol: str = Field(..., description="Ticker or asset code")
    value: float = Field(..., gt=0, description="Asset market value")
    volatility: float = Field(default=0.15, ge=0.0, le=2.0)
    score: float = Field(default=0.5, ge=0.0, le=1.0)


class OptimizationRequest(BaseModel):
    assets: List[Asset] = Field(..., min_length=1)
    iterations: int = Field(default=3, ge=1, le=20)
    target_return: float = Field(default=0.0, ge=-1.0, le=1.0)


class OptimizationResponse(BaseModel):
    status: str
    best_asset: str
    best_score: float
    mean_score: float
    winner_index: int
    metrics: Dict[str, Any]


class PortfolioRiskRequest(BaseModel):
    portfolio: List[Asset] = Field(..., min_length=1)
    stress_scenario: str = Field(default="baseline")


class HealthResponse(BaseModel):
    status: str
    ai_available: bool
    quantum_available: bool
    environment: str
    version: str = "1.1.0"
