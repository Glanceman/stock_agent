"""Agent tools and market-data helpers for the stock trading expert."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, cast

import pandas as pd
import yfinance as yf
from ddgs import DDGS
from pydantic_ai import Agent, RunContext

from stock_agent.deps import StockAgentDeps
from stock_agent.models import (
    IndicatorValue,
    NewsItem,
    PriceBar,
    PriceHistory,
    TechnicalReport,
    TradeAnalysis,
)


class StockDataError(RuntimeError):
    """Raised when a data provider cannot return usable market data."""


def _clean_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _to_float(value: Any) -> float:
    return float(cast(float, value))


def _to_optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(_to_float(value), 4)


def _to_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(float(cast(float, value)))


def _date_string(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def _normalize_history_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [list(map(str, frame.columns.get_level_values(i))) for i in range(frame.columns.nlevels)]
        if "Close" in levels[0]:
            frame = frame.droplevel(1, axis=1)
        elif ticker in levels[0]:
            frame = frame.xs(ticker, axis=1, level=0)
        elif ticker in levels[1]:
            frame = frame.xs(ticker, axis=1, level=1)

    frame = frame.rename(columns={str(column): str(column).title() for column in frame.columns})
    if "Close" not in frame.columns:
        raise StockDataError(f"No close prices were returned for {ticker}.")
    return frame.dropna(subset=["Close"])


def _download_history_sync(ticker: str, period: str, interval: str) -> pd.DataFrame:
    frame = cast(
        pd.DataFrame,
        yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        ),
    )
    frame = _normalize_history_frame(frame, ticker)
    if frame.empty:
        raise StockDataError(f"No price history was returned for {ticker}. Check the ticker symbol.")
    return frame


def _summary_sentence(text: str, fallback: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        cleaned = fallback
    sentence = cleaned.split(". ", maxsplit=1)[0].strip()
    return sentence if sentence.endswith(".") else f"{sentence}."


def _fetch_news_sync(ticker: str, max_results: int) -> list[NewsItem]:
    query = f"{ticker} stock latest earnings analyst rating market news"
    with DDGS() as ddgs:
        raw_results = list(ddgs.news(query, max_results=max(1, min(max_results, 10))))

    items: list[NewsItem] = []
    for raw in raw_results:
        row = cast(dict[str, Any], raw)
        headline = str(row.get("title") or "Untitled news item")
        body = str(row.get("body") or "")
        items.append(
            NewsItem(
                headline=headline,
                source=str(row.get("source") or "Unknown source"),
                date=str(row.get("date") or "Unknown date"),
                url=str(row.get("url") or row.get("href") or ""),
                summary=_summary_sentence(body, headline),
            )
        )
    return items


async def fetch_stock_news(ticker: str, max_results: int = 5) -> list[NewsItem]:
    """Fetch recent stock news for a ticker using DuckDuckGo Search."""

    cleaned_ticker = _clean_ticker(ticker)
    if not cleaned_ticker:
        raise StockDataError("Please provide a ticker symbol.")
    try:
        return await asyncio.to_thread(_fetch_news_sync, cleaned_ticker, max_results)
    except Exception as exc:  # noqa: BLE001 - provider failures should be user-friendly.
        raise StockDataError(f"Could not fetch recent news for {cleaned_ticker}: {exc}") from exc


async def fetch_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> PriceHistory:
    """Download and summarize OHLCV price history from Yahoo Finance."""

    cleaned_ticker = _clean_ticker(ticker)
    if not cleaned_ticker:
        raise StockDataError("Please provide a ticker symbol.")

    frame = await asyncio.to_thread(_download_history_sync, cleaned_ticker, period, interval)
    first_close = _to_float(frame["Close"].iloc[0])
    latest_close = _to_float(frame["Close"].iloc[-1])
    percent_change = ((latest_close - first_close) / first_close) * 100 if first_close else 0.0

    recent_bars: list[PriceBar] = []
    for index, row in frame.tail(5).iterrows():
        recent_bars.append(
            PriceBar(
                date=_date_string(index),
                open=round(_to_float(row.get("Open", row["Close"])), 4),
                high=round(_to_float(row.get("High", row["Close"])), 4),
                low=round(_to_float(row.get("Low", row["Close"])), 4),
                close=round(_to_float(row["Close"]), 4),
                volume=_to_int(row.get("Volume", 0)),
            )
        )

    return PriceHistory(
        ticker=cleaned_ticker,
        period=period,
        interval=interval,
        start_date=_date_string(frame.index[0]),
        end_date=_date_string(frame.index[-1]),
        rows=len(frame),
        latest_close=round(latest_close, 4),
        period_high=round(_to_float(frame["High"].max() if "High" in frame else frame["Close"].max()), 4),
        period_low=round(_to_float(frame["Low"].min() if "Low" in frame else frame["Close"].min()), 4),
        latest_volume=_to_int(frame["Volume"].iloc[-1] if "Volume" in frame else 0),
        percent_change=round(percent_change, 2),
        recent_bars=recent_bars,
    )


def _indicator(name: str, value: float | None, interpretation: str) -> IndicatorValue:
    return IndicatorValue(name=name, value=value, interpretation=interpretation)


def _rsi_interpretation(value: float | None) -> str:
    if value is None:
        return "Not enough price history to compute RSI-14."
    if value >= 70:
        return f"RSI {value:.1f} -> overbought momentum."
    if value <= 30:
        return f"RSI {value:.1f} -> oversold momentum."
    return f"RSI {value:.1f} -> neutral momentum."


def _sma_interpretation(close: float, name: str, value: float | None) -> str:
    if value is None:
        return f"Not enough data to compute {name}."
    relation = "above" if close >= value else "below"
    return f"Price is {relation} {name} ({value:.2f})."


def _classify_trend(
    close: float,
    sma_20: float | None,
    sma_50: float | None,
    macd: float | None,
    macd_signal: float | None,
) -> Literal["bullish", "bearish", "neutral"]:
    if sma_20 is not None and sma_50 is not None and macd is not None and macd_signal is not None:
        if close > sma_20 > sma_50 and macd > macd_signal:
            return "bullish"
        if close < sma_20 < sma_50 and macd < macd_signal:
            return "bearish"
    return "neutral"


async def build_technical_report(ticker: str) -> TechnicalReport:
    """Compute moving averages, RSI, MACD, Bollinger Bands, and trend context."""

    cleaned_ticker = _clean_ticker(ticker)
    if not cleaned_ticker:
        raise StockDataError("Please provide a ticker symbol.")

    frame = await asyncio.to_thread(_download_history_sync, cleaned_ticker, "1y", "1d")
    close = cast(pd.Series, frame["Close"].astype("float64"))
    latest_close = _to_float(close.iloc[-1])

    sma_20_series = close.rolling(window=20, min_periods=20).mean()
    sma_50_series = close.rolling(window=50, min_periods=50).mean()
    sma_200_series = close.rolling(window=200, min_periods=200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.where(loss != 0)
    rsi_14_series = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_series = ema_12 - ema_26
    signal_series = macd_series.ewm(span=9, adjust=False).mean()
    histogram_series = macd_series - signal_series

    bollinger_middle_series = close.rolling(window=20, min_periods=20).mean()
    bollinger_std_series = close.rolling(window=20, min_periods=20).std()
    bollinger_upper_series = bollinger_middle_series + (2 * bollinger_std_series)
    bollinger_lower_series = bollinger_middle_series - (2 * bollinger_std_series)

    sma_20 = _to_optional_float(sma_20_series.iloc[-1])
    sma_50 = _to_optional_float(sma_50_series.iloc[-1])
    sma_200 = _to_optional_float(sma_200_series.iloc[-1])
    rsi_14 = _to_optional_float(rsi_14_series.iloc[-1])
    macd = _to_optional_float(macd_series.iloc[-1])
    macd_signal = _to_optional_float(signal_series.iloc[-1])
    macd_histogram = _to_optional_float(histogram_series.iloc[-1])
    bollinger_upper = _to_optional_float(bollinger_upper_series.iloc[-1])
    bollinger_middle = _to_optional_float(bollinger_middle_series.iloc[-1])
    bollinger_lower = _to_optional_float(bollinger_lower_series.iloc[-1])

    trend = _classify_trend(latest_close, sma_20, sma_50, macd, macd_signal)
    macd_interpretation = (
        "MACD is above its signal line, supporting positive momentum."
        if macd is not None and macd_signal is not None and macd > macd_signal
        else "MACD is below or near its signal line, showing weak or negative momentum."
    )
    bollinger_interpretation = (
        "Price is near the upper Bollinger Band, which can indicate stretched upside."
        if bollinger_upper is not None and latest_close >= bollinger_upper * 0.98
        else "Price is not pressing the upper Bollinger Band."
    )

    return TechnicalReport(
        ticker=cleaned_ticker,
        as_of=_date_string(frame.index[-1]),
        close=round(latest_close, 4),
        sma_20=_indicator("SMA-20", sma_20, _sma_interpretation(latest_close, "SMA-20", sma_20)),
        sma_50=_indicator("SMA-50", sma_50, _sma_interpretation(latest_close, "SMA-50", sma_50)),
        sma_200=_indicator("SMA-200", sma_200, _sma_interpretation(latest_close, "SMA-200", sma_200)),
        rsi_14=_indicator("RSI-14", rsi_14, _rsi_interpretation(rsi_14)),
        macd=_indicator("MACD", macd, macd_interpretation),
        macd_signal=_indicator("MACD Signal", macd_signal, macd_interpretation),
        macd_histogram=_indicator("MACD Histogram", macd_histogram, macd_interpretation),
        bollinger_upper=_indicator("Bollinger Upper", bollinger_upper, bollinger_interpretation),
        bollinger_middle=_indicator("Bollinger Middle", bollinger_middle, bollinger_interpretation),
        bollinger_lower=_indicator("Bollinger Lower", bollinger_lower, bollinger_interpretation),
        trend=trend,
        interpretation=(
            f"{cleaned_ticker} is classified as {trend} based on moving-average alignment, "
            f"MACD confirmation, RSI context, and Bollinger Band position as of {_date_string(frame.index[-1])}."
        ),
    )


def register_tools(agent: Agent[StockAgentDeps, TradeAnalysis]) -> None:
    """Register all Pydantic AI tools on the provided stock analysis agent."""

    @agent.tool
    async def think(ctx: RunContext[StockAgentDeps], reasoning: str, next_action: str) -> str:
        """Persist a scratchpad reasoning step and state the next planned action."""

        ctx.deps.count_tool_call()
        ctx.deps.emit_progress("Thinking through the next research step...")
        return ctx.deps.record_reasoning(reasoning, next_action)

    @agent.tool
    async def search_stock_news(
        ctx: RunContext[StockAgentDeps],
        ticker: str,
        max_results: int = 5,
    ) -> list[NewsItem]:
        """Search recent market news for a ticker."""

        ctx.deps.count_tool_call()
        cleaned_ticker = _clean_ticker(ticker)
        ctx.deps.emit_progress(f"🔍 Searching news for {cleaned_ticker}...")
        return await fetch_stock_news(cleaned_ticker, max_results)

    @agent.tool
    async def get_price_history(
        ctx: RunContext[StockAgentDeps],
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> PriceHistory:
        """Download OHLCV price history for a ticker."""

        ctx.deps.count_tool_call()
        cleaned_ticker = _clean_ticker(ticker)
        ctx.deps.emit_progress(f"📈 Downloading {period} {interval} price history for {cleaned_ticker}...")
        return await fetch_price_history(cleaned_ticker, period, interval)

    @agent.tool
    async def compute_technicals(ctx: RunContext[StockAgentDeps], ticker: str) -> TechnicalReport:
        """Compute the required technical indicators for a ticker."""

        ctx.deps.count_tool_call()
        cleaned_ticker = _clean_ticker(ticker)
        ctx.deps.emit_progress(f"📊 Computing RSI, MACD, moving averages, and Bollinger Bands for {cleaned_ticker}...")
        return await build_technical_report(cleaned_ticker)
