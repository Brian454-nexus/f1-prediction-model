"""
F1 APEX — Autonomous News Fetcher
================================
Uses DuckDuckGo to pull targeted F1 news for teams and session context.
"""

from duckduckgo_search import DDGS
from utils.logger import get_logger

logger = get_logger("APEX.NewsFetcher")

def fetch_f1_news(scope: str, session_id: str = "GP") -> str:
    """
    Performs an anonymous search for F1 news based on the scope.
    Scope can be: "All Teams", "Top 5", or a specific team name (e.g. "Mercedes").
    """
    query = f"F1 {scope} technical updates news 2026 Suzuka {session_id}"
    logger.info(f"Searching web for: '{query}'")

    results_text = ""
    try:
        with DDGS() as ddgs:
            # We only need the top 4-5 results for token efficiency
            results = ddgs.text(query, max_results=5)
            for r in results:
                results_text += f"- {r['title']}: {r['body'][:200]}...\n"
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return "No recent web news found due to search timeout."

    if not results_text:
        return "No specific news found for this scope currently."

    return results_text
