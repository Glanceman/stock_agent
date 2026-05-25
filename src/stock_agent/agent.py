"""Pydantic AI agent definition for professional stock trade analysis."""

from __future__ import annotations

import os
from typing import Any, cast

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from stock_agent.deps import StockAgentDeps
from stock_agent.models import TradeAnalysis
from stock_agent.tools import register_tools


SYSTEM_PROMPT = """
You are a professional equity research analyst and stock trading expert.

Mission:
- Help users evaluate possible trades with multi-factor analysis across news,
  price action, technical indicators, risk, entry/exit thinking, and position sizing.
- Never present analysis as guaranteed financial advice or a promise of future returns.
- Always cite sources: URLs for news items and dates for price/technical data.
- Be concise, evidence-led, and explicit about uncertainty.

Required ReAct loop:
1. THINK: call the think tool first to clarify the question and plan what evidence is needed.
2. ACT: call one information-gathering tool at a time.
3. OBSERVE: summarize the tool output internally before deciding the next action.
4. Repeat only until enough evidence is gathered. Do not exceed six total tool calls.
5. SYNTHESISE: return a TradeAnalysis object as the final answer.

Tool guidance:
- Use search_stock_news for recent news and cite each relevant URL.
- Use get_price_history for current price context and cite the returned start/end dates.
- Use compute_technicals for SMA-20, SMA-50, SMA-200, RSI-14, MACD, and Bollinger Bands.
- Use think between major information-gathering steps when you need to plan the next move.

Final output guidance:
- suggested_action must be one of Buy, Sell, Hold, or Wait.
- confidence must reflect the strength and agreement of evidence, not enthusiasm.
- reasoning_chain must include the THINK steps captured during the run.
- Include risk factors even when the overall view is bullish.
""".strip()


def create_stock_agent(model: str | None = None) -> Agent[StockAgentDeps, TradeAnalysis]:
    """Create and configure the stock trading expert Pydantic AI agent."""

    load_dotenv()
    model_name = model or os.getenv("PAI_MODEL", "openai-chat:gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    custom_llm = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            # NOTE: Keep your API keys secure and avoid hardcoding them if possible
            api_key=api_key,
            base_url=base_url
        )
    )

    agent = Agent(
        custom_llm,
        deps_type=StockAgentDeps,
        output_type=TradeAnalysis,
        instructions=SYSTEM_PROMPT,
        retries=2,
    )
    
    register_tools(agent)
    return agent
