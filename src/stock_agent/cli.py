"""Rich command-line REPL for the stock trading expert agent."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from stock_agent.agent import create_stock_agent
from stock_agent.deps import StockAgentDeps
from stock_agent.models import NewsItem, TechnicalReport, TradeAnalysis
from stock_agent.tools import StockDataError, build_technical_report, fetch_stock_news


DISCLAIMER = (
    "This tool provides informational market analysis only. It is not financial advice, "
    "and it cannot guarantee outcomes. Always do your own due diligence."
)


def render_banner(console: Console) -> None:
    """Render the startup banner and disclaimer."""

    console.print(
        Panel.fit(
            "[bold green]Stock Trading Expert Agent[/bold green]\n"
            "Pydantic AI + DuckDuckGo news + Yahoo Finance technical analysis\n\n"
            f"[yellow]{DISCLAIMER}[/yellow]",
            border_style="green",
        )
    )


def render_help(console: Console) -> None:
    """Render REPL command help."""

    console.print(
        Markdown(
            """
### Commands

- `/help` - show this usage guide
- `/analyze AAPL` - run a full trade analysis
- `/news TSLA` - show recent ticker news only
- `/technicals MSFT` - compute technical indicators only
- `/quit` - exit

You can also type a natural-language question, such as:
`Should I wait for a pullback before buying NVDA?`
"""
        )
    )


def render_news(console: Console, ticker: str, items: list[NewsItem]) -> None:
    """Render news results in a Rich table."""

    table = Table(title=f"Recent News: {ticker.upper()}", box=box.SIMPLE_HEAVY)
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Source", style="magenta")
    table.add_column("Headline")
    table.add_column("URL", overflow="fold")

    if not items:
        console.print(f"[yellow]No recent news found for {ticker.upper()}.[/yellow]")
        return

    for item in items:
        table.add_row(item.date, item.source, f"{item.headline}\n[dim]{item.summary}[/dim]", str(item.url))
    console.print(table)


def render_technical_report(console: Console, report: TechnicalReport) -> None:
    """Render a technical-analysis report in a Rich table."""

    table = Table(title=f"Technicals: {report.ticker} as of {report.as_of}", box=box.SIMPLE_HEAVY)
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation")

    indicators = [
        report.sma_20,
        report.sma_50,
        report.sma_200,
        report.rsi_14,
        report.macd,
        report.macd_signal,
        report.macd_histogram,
        report.bollinger_upper,
        report.bollinger_middle,
        report.bollinger_lower,
    ]
    for indicator in indicators:
        value = "n/a" if indicator.value is None else f"{indicator.value:.2f}"
        table.add_row(indicator.name, value, indicator.interpretation)

    console.print(table)
    console.print(f"[bold]Trend:[/bold] {report.trend.upper()} - {report.interpretation}")
    console.print(f"[dim]Source: {report.source}[/dim]")


def render_trade_analysis(console: Console, analysis: TradeAnalysis) -> None:
    """Render the final structured trade analysis."""

    summary = Table(title=f"Trade Analysis: {analysis.ticker}", box=box.ROUNDED)
    summary.add_column("Field", style="cyan", no_wrap=True)
    summary.add_column("Value")
    summary.add_row("Sentiment", analysis.sentiment)
    summary.add_row("Suggested Action", analysis.suggested_action)
    summary.add_row("Confidence", f"{analysis.confidence:.0%}")
    summary.add_row("Technical Outlook", analysis.technical_outlook)
    console.print(summary)

    report = [
        "### News Highlights",
        *[f"- {item}" for item in analysis.news_highlights],
        "",
        "### Risk Factors",
        *[f"- {item}" for item in analysis.risk_factors],
        "",
        "### Reasoning Chain",
        *[f"- {item}" for item in analysis.reasoning_chain],
        "",
        "### Disclaimer",
        analysis.disclaimer,
    ]
    console.print(Markdown("\n".join(report)))


def _extract_output(result: Any) -> TradeAnalysis:
    output = getattr(result, "output", None)
    if output is None:
        output = getattr(result, "data", None)
    return output if isinstance(output, TradeAnalysis) else TradeAnalysis.model_validate(output)


async def run_agent_analysis(console: Console, user_prompt: str) -> TradeAnalysis:
    """Run the Pydantic AI agent and return a structured trade analysis."""

    agent = create_stock_agent()

    def progress(message: str) -> None:
        console.print(Text(message, style="cyan"))

    async with httpx.AsyncClient(timeout=20.0) as client:
        deps = StockAgentDeps(http_client=client, progress_callback=progress)
        result = await agent.run(user_prompt, deps=deps)
        analysis = _extract_output(result)

    if deps.reasoning_chain:
        analysis = analysis.model_copy(update={"reasoning_chain": deps.reasoning_chain})
    render_trade_analysis(console, analysis)
    return analysis


def _command_arg(command_line: str) -> str:
    parts = command_line.split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


async def handle_command(console: Console, command_line: str) -> bool:
    """Handle one REPL command and return whether the loop should continue."""

    command = command_line.split(maxsplit=1)[0].lower()
    arg = _command_arg(command_line)

    if command in {"/quit", "/exit"}:
        console.print("[green]Goodbye.[/green]")
        return False
    if command == "/help":
        render_help(console)
        return True
    if command == "/news":
        if not arg:
            console.print("[yellow]Usage: /news TSLA[/yellow]")
            return True
        console.print(Text(f"🔍 Searching news for {arg.upper()}...", style="cyan"))
        render_news(console, arg, await fetch_stock_news(arg))
        return True
    if command == "/technicals":
        if not arg:
            console.print("[yellow]Usage: /technicals MSFT[/yellow]")
            return True
        console.print(Text(f"📊 Computing technicals for {arg.upper()}...", style="cyan"))
        render_technical_report(console, await build_technical_report(arg))
        return True
    if command == "/analyze":
        if not arg:
            console.print("[yellow]Usage: /analyze AAPL[/yellow]")
            return True
        await run_agent_analysis(
            console,
            f"Analyze {arg.upper()} for a potential trade using news, price history, and technical indicators.",
        )
        return True

    console.print("[yellow]Unknown command. Type /help for usage.[/yellow]")
    return True


async def run_repl() -> None:
    """Run the interactive Rich REPL."""

    console = Console()
    render_banner(console)
    render_help(console)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]stock-agent[/bold cyan]").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = await handle_command(console, user_input)
            else:
                should_continue = True
                await run_agent_analysis(console, user_input)

            if not should_continue:
                break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /quit to exit or enter another request.[/yellow]")
        except (StockDataError, httpx.HTTPError, ValueError) as exc:
            console.print(f"[red]Unable to complete request:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001 - CLI must fail without tracebacks.
            console.print(f"[red]The agent could not complete that request:[/red] {exc}")


def main() -> None:
    """Console-script entry point."""

    try:
        asyncio.run(run_repl())
    except KeyboardInterrupt:
        Console().print("\n[yellow]Goodbye.[/yellow]")
