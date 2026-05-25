"""Pydantic schemas used by the stock trading expert agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class NewsItem(BaseModel):
    """A single recent news item about a publicly traded company."""

    headline: str
    source: str
    date: str
    url: HttpUrl | str
    summary: str


class PriceBar(BaseModel):
    """A compact OHLCV bar used to cite recent market data."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceHistory(BaseModel):
    """A serializable summary of a ticker's downloaded price history."""

    ticker: str
    period: str
    interval: str
    start_date: str
    end_date: str
    rows: int
    latest_close: float
    period_high: float
    period_low: float
    latest_volume: int
    percent_change: float
    recent_bars: list[PriceBar]
    source: str = "Yahoo Finance via yfinance"


class IndicatorValue(BaseModel):
    """A numeric indicator reading with plain-English context."""

    name: str
    value: float | None
    interpretation: str


class TechnicalReport(BaseModel):
    """A structured technical-analysis report for one ticker."""

    ticker: str
    as_of: str
    close: float
    sma_20: IndicatorValue
    sma_50: IndicatorValue
    sma_200: IndicatorValue
    rsi_14: IndicatorValue
    macd: IndicatorValue
    macd_signal: IndicatorValue
    macd_histogram: IndicatorValue
    bollinger_upper: IndicatorValue
    bollinger_middle: IndicatorValue
    bollinger_lower: IndicatorValue
    trend: Literal["bullish", "bearish", "neutral"]
    interpretation: str
    source: str = "Yahoo Finance via yfinance; indicators computed with pandas"


class TradeAnalysis(BaseModel):
    """The agent's final structured trading analysis."""

    ticker: str
    sentiment: Literal["Bullish", "Bearish", "Neutral"]
    technical_outlook: str
    news_highlights: list[str]
    risk_factors: list[str]
    suggested_action: Literal["Buy", "Sell", "Hold", "Wait"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_chain: list[str]
    disclaimer: str = (
        "This is informational analysis, not financial advice. "
        "Always do your own due diligence."
    )
