# Stock Trading Expert Agent

A production-quality command-line equity research assistant built with
Pydantic AI, `uv`, DuckDuckGo Search, Yahoo Finance, pandas, and Rich.

The agent behaves like a professional equity research analyst: it gathers
recent news, downloads price history, computes technical indicators, records
explicit ReAct-style reasoning steps with a `think` tool, and returns a
structured `TradeAnalysis` report.

> This is informational analysis, not financial advice. Always do your own due diligence.

Python 3.11, 3.12, or 3.13 is recommended. The current dependency set includes
compiled numeric packages whose Windows wheels are not consistently available
for Python 3.14 yet.

## Install

```bash
cd stock_agent
uv sync
cp .env.example .env
```

Edit `.env` and set the model plus API key for your Pydantic AI provider:

```bash
PAI_MODEL=openai-chat:gpt-4o-mini
OPENAI_API_KEY=sk-your-openai-key
```

Run the REPL:

```bash
uv run stock_agent
```

You can also run it as a module:

```bash
uv run python -m stock_agent
```

## Exact dependency commands

If recreating the project from an empty folder, these are the runtime dependency
commands:

```bash
uv add pydantic-ai duckduckgo-search yfinance pandas rich python-dotenv httpx
uv add "pandas-ta; python_version >= '3.12' and python_version < '3.14'"
uv add --dev mypy
```

The agent computes indicators manually with pandas, so it remains usable on
Python 3.11. The `pandas-ta` package currently resolves on Python 3.12+ in the
available index, and its `numba/llvmlite` chain does not currently build on the
active Python 3.14 environment, so it is declared with a Python-version marker.

## Commands

- `/help` - show usage
- `/analyze AAPL` - run a full trade analysis
- `/news TSLA` - fetch recent news only
- `/technicals MSFT` - compute technical indicators only
- `/quit` - exit

Free-form questions are sent to the agent, for example:

```text
Should I wait for a pullback before buying NVDA?
```

## Example Conversations

### Full Analysis

```text
stock-agent> /analyze NVDA
Thinking through the next research step...
🔍 Searching news for NVDA...
Thinking through the next research step...
📈 Downloading 6mo 1d price history for NVDA...
📊 Computing RSI, MACD, moving averages, and Bollinger Bands for NVDA...

Trade Analysis: NVDA
Sentiment: Bullish
Suggested Action: Wait
Confidence: 72%
Technical Outlook: NVDA remains in an uptrend, but RSI and Bollinger context
suggest chasing the move may carry short-term pullback risk.

News Highlights
- Recent AI data-center demand coverage remains supportive; source URL cited by the agent.
- Earnings and guidance commentary should be monitored for margin and supply-chain risk.

Risk Factors
- Valuation sensitivity if AI growth expectations cool.
- Broad market weakness can overwhelm strong single-name fundamentals.
- Semiconductor export controls and supply constraints remain headline risks.
```

### News Only

```text
stock-agent> /news TSLA
🔍 Searching news for TSLA...

Recent News: TSLA
Date          Source          Headline
2026-05-24    Example News    Tesla shares move after delivery and pricing headlines...
```

### Technicals Only

```text
stock-agent> /technicals MSFT
📊 Computing technicals for MSFT...

Technicals: MSFT as of 2026-05-22
Indicator          Value       Interpretation
SMA-20             428.15      Price is above SMA-20 (428.15).
RSI-14              58.30      RSI 58.3 -> neutral momentum.
MACD                 3.72      MACD is above its signal line, supporting positive momentum.
Trend: BULLISH
```

## Sample `/analyze NVDA` Terminal Session

```text
$ uv run stock_agent
╭────────────────────────────────────────────╮
│ Stock Trading Expert Agent                 │
│ Pydantic AI + DuckDuckGo news + Yahoo ...  │
│                                            │
│ This tool provides informational market... │
╰────────────────────────────────────────────╯

stock-agent> /analyze NVDA
Thinking through the next research step...
🔍 Searching news for NVDA...
Thinking through the next research step...
📈 Downloading 6mo 1d price history for NVDA...
📊 Computing RSI, MACD, moving averages, and Bollinger Bands for NVDA...

Trade Analysis: NVDA
┌───────────────────┬──────────────────────────────────────────────────────┐
│ Field             │ Value                                                │
├───────────────────┼──────────────────────────────────────────────────────┤
│ Sentiment         │ Bullish                                              │
│ Suggested Action  │ Wait                                                 │
│ Confidence        │ 72%                                                  │
│ Technical Outlook │ Uptrend intact, but short-term entries need discipline│
└───────────────────┴──────────────────────────────────────────────────────┘

News Highlights
- AI infrastructure demand remains a key bullish driver. Source: https://...
- Recent analyst coverage is constructive but valuation-sensitive. Source: https://...

Risk Factors
- High expectations increase downside risk on any guidance miss.
- A broad market drawdown could pressure high-multiple semiconductor names.

Reasoning Chain
- Clarify whether NVDA has supportive news, price trend, and technical confirmation.
  Next action: Search recent news.
- News appears supportive but valuation-sensitive. Next action: Check price history.

Disclaimer
This is informational analysis, not financial advice. Always do your own due diligence.
```

Live output will vary because news and market data are fetched at runtime.

## Development

```bash
uv run mypy
```

## Extension Ideas

- Portfolio tracking with position-level risk and exposure limits.
- Telegram or Discord bot wrapper for watchlist alerts.
- Back-testing with `vectorbt` or `backtrader`.
- Earnings-calendar and SEC filing tools.
- Persistent SQLite watchlists and cached news snapshots.
