"""
F1 APEX — AI Session Reporter CLI
================================
Generates high-fidelity narratives by synthesizing telemetry with live news.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from data.session_summarizer import summarize_session
from utils.llm_client import F1ReportClient
from utils.news_fetcher import fetch_f1_news
from utils.logger import get_logger

logger = get_logger("APEX.Reporter")
console = Console()

# Mapping Suzuka 2026 context (Round 3)
YEAR = 2026
ROUND_ID = 3
RAW_TELEMETRY_DIR = Path("data/raw/telemetry")

def get_parquet_path(session_id: str) -> Path:
    """Matches the naming convention in data_pipeline.py"""
    return RAW_TELEMETRY_DIR / f"laps_{YEAR}_R{ROUND_ID:02d}_{session_id}.parquet"

async def generate_session_report(session_id: str, scope: str = "All Teams", manual_news: str | None = None):
    """
    Main flow: Summarize Telemetry -> Fetch News -> Generate AI Narrative.
    """
    parquet_file = get_parquet_path(session_id)
    if not parquet_file.exists():
        console.print(f"[bold red]Error:[/bold red] Telemetry for {session_id} not found at {parquet_file}")
        return

    # 1. Summarize Telemetry
    telemetry_summary = summarize_session(str(parquet_file), session_id)
    
    # 2. Fetch News (Autonomous or Manual override)
    if manual_news:
        news = manual_news
    else:
        with console.status(f"[bold cyan]Searching for live {scope} news...", spinner="earth"):
            news = fetch_f1_news(scope, session_id)

    # 3. Setup LLM Persona & Prompt
    client = F1ReportClient()
    system_prompt = (
        "You are the Lead Race Engineer for the F1 APEX Strategy team. "
        "Your tone is professional, technical, and analytical. You sound like an insider. "
        "Analyze the provided session telemetry and news to give a high-level narrative. "
        "Focus on: PACE TRENDS, RELIABILITY CONCERNS, and UPGRADE IMPACT. "
        "Be concise but insightful. Keep it under 400 words. Format with Markdown."
    )
    
    user_prompt = (
        f"### SESSION: {session_id} (Suzuka, GP 2026)\n\n"
        f"#### SCOPE: {scope}\n\n"
        f"#### TELEMETRY SUMMARY (High-Density JSON):\n{telemetry_summary}\n\n"
        f"#### LIVE NEWS BRIEF:\n{news}\n\n"
        "Generate the Engineer's Report now."
    )

    with console.status(f"[bold green]Claude is analyzing {session_id} data...", spinner="dots"):
        report_md = await client.generate_report(system_prompt, user_prompt)

    # 4. Display Result
    console.print(Panel(Markdown(report_md), title=f"📊 F1 APEX — AI ENGINEER REPORT: {session_id}", border_style="cyan"))

async def generate_weekend_summary(scope: str):
    """
    Aggregates all available session data (FP1-Q) into a single master report.
    """
    sessions = ["FP1", "FP2", "FP3", "Q"]
    all_telemetry = {}
    
    for s in sessions:
        p = get_parquet_path(s)
        if p.exists():
            all_telemetry[s] = summarize_session(str(p), s)

    if not all_telemetry:
        console.print("[bold red]Error:[/bold red] No session data found to summarize.")
        return

    with console.status("[bold cyan]Fetching full weekend news...", spinner="earth"):
        news = fetch_f1_news(f"{scope} weekend recap", "Full Weekend")

    client = F1ReportClient()
    system_prompt = "You are the Head of Strategy. Provide a high-level weekend summary based on all sessions. Identify the 'Momentum Shift' for the race."
    user_prompt = f"WEEKEND DATA:\n{all_telemetry}\n\nNEWS:\n{news}\n\nGenerate the Weekend Strategy Brief."

    with console.status("[bold green]Claude is synthesizing the weekend...", spinner="dots"):
        report_md = await client.generate_report(system_prompt, user_prompt)

    console.print(Panel(Markdown(report_md), title="🏁 F1 APEX — WEEKEND STRATEGY BRIEF", border_style="gold1"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 APEX AI Reporter")
    parser.add_argument("--session", choices=["FP1", "FP2", "FP3", "Q"], help="Session to report on")
    parser.add_argument("--summary", action="store_true", help="Generate a full weekend summary (FP1-Q)")
    parser.add_argument("--scope", type=str, default="All Teams", help="News scope: 'All Teams', 'Top 5', or a specific team name")
    parser.add_argument("--news", type=str, help="Optional manual news/upgrade details override")
    args = parser.parse_args()

    if args.summary:
        asyncio.run(generate_weekend_summary(scope=args.scope))
    elif args.session:
        asyncio.run(generate_session_report(args.session, scope=args.scope, manual_news=args.news))
    else:
        parser.print_help()

