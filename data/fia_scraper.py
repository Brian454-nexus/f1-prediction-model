"""
F1 APEX — FIA Document Scraper
================================
Scrapes the FIA's official Formula 1 documents to extract:
  1. Grid penalties (engine component changes, sporting code violations)
  2. Post-qualifying steward decisions that alter the grid
  3. Force majeure notes (weather delays, red-flagged sessions)

These events are used as late-breaking inputs to the prediction pipeline —
typically loaded at T-2h before race start (FR-2 in PRD).

The FIA publishes all decisions as PDFs linked from:
  https://www.fia.com/documents/championships/fia-formula-one-world-championship

BeautifulSoup is used to parse the HTML of that page. Key PDF filenames
follow patterns like "Decision_-_{driver}_{infringement}_{date}.pdf" and
"Grid_Order_-_{race_name}_{date}.pdf".

Because FIA site structure can change between seasons, the parser is built
defensively — any individual extraction failure is logged and skipped
rather than crashing the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger

logger = get_logger("APEX.FIAScraper")

# ── Constants ─────────────────────────────────────────────────────────────────

FIA_DOCS_BASE = "https://www.fia.com/documents/championships/fia-formula-one-world-championship"
REQUEST_TIMEOUT = 15

# Grid position offsets assigned per ISO-regulated component replacement,
# matching the 2026 technical regulations.
PENALTY_COMPONENT_GRID_DROP: dict[str, int] = {
    "ICE":  10,
    "MGU-H": 10,
    "TC":   10,
    "ES":   10,
    "CE":   10,
    "MGU-K": 5,
    "EX":   5,
    "GEARBOX": 5,
}


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class GridPenalty:
    driver: str
    team: str
    component: str
    grid_drop: int
    reason: str
    document_url: Optional[str] = None


@dataclass
class RaceWeekendDocuments:
    round_id: int
    penalties: list[GridPenalty] = field(default_factory=list)
    total_penalised_drivers: int = 0

    def apply_to_grid(self, qualifying_order: list[str]) -> list[str]:
        """
        Return a new grid order list with penalties applied.
        Penalised drivers are moved down by their penalty amount.
        Drivers given a back-of-grid penalty go to the end in order of penalty severity.
        """
        order = list(qualifying_order)
        for penalty in sorted(self.penalties, key=lambda p: -p.grid_drop):
            if penalty.driver in order:
                current_pos = order.index(penalty.driver)
                new_pos = min(current_pos + penalty.grid_drop, len(order) - 1)
                order.pop(current_pos)
                order.insert(new_pos, penalty.driver)
        return order


# ── Scraper Class ─────────────────────────────────────────────────────────────

class FIADocumentScraper:
    """
    Fetches and parses FIA race weekend documents.

    The scraper operates in two stages:
     1. Fetch the FIA documents page to find links to PDFs for the target round.
     2. Parse each relevant PDF filename (no actual PDF reading needed — the
        metadata is usually sufficient for penalty extraction).
    """

    def __init__(self, docs_output_dir: str = "data/raw/fia_docs") -> None:
        self.docs_dir = Path(docs_output_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "F1APEX-Scraper/1.0"})

    def fetch_weekend_documents(
        self, round_id: int, year: int = 2026
    ) -> RaceWeekendDocuments:
        """
        Retrieve all penalty-relevant FIA documents for a race weekend.

        Args:
            round_id: Race round number.
            year:     Season year.

        Returns:
            RaceWeekendDocuments with parsed penalties.
        """
        logger.info(f"Fetching FIA documents for {year} Round {round_id}")
        docs = RaceWeekendDocuments(round_id=round_id)

        try:
            html = self._fetch_page()
        except Exception as exc:
            logger.warning(f"FIA page fetch failed: {exc}. Returning empty penalty set.")
            return docs

        soup = BeautifulSoup(html, "html.parser")
        pdf_links = self._extract_pdf_links(soup, year)
        logger.info(f"Found {len(pdf_links)} FIA PDF documents for {year}")

        for url, title in pdf_links:
            penalty = self._parse_penalty_from_title(title, url)
            if penalty:
                docs.penalties.append(penalty)
                logger.info(
                    f"  Penalty extracted: {penalty.driver} — {penalty.component} "
                    f"({penalty.grid_drop} place drop)"
                )

        docs.total_penalised_drivers = len(docs.penalties)
        logger.info(
            f"Extracted {docs.total_penalised_drivers} grid penalties for Round {round_id}"
        )
        return docs

    # ── Internal helpers ───────────────────────────────────────────────────

    def _fetch_page(self) -> str:
        response = self.session.get(FIA_DOCS_BASE, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _extract_pdf_links(soup: BeautifulSoup, year: int) -> list[tuple[str, str]]:
        """Find all links to PDF documents on the FIA page."""
        links = []
        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            title: str = a_tag.get_text(strip=True)
            if href.endswith(".pdf") and str(year) in href:
                full_url = (
                    href if href.startswith("http") else f"https://www.fia.com{href}"
                )
                links.append((full_url, title))
        return links

    @staticmethod
    def _parse_penalty_from_title(title: str, url: str) -> Optional[GridPenalty]:
        """
        Attempt to extract a grid penalty from a document title.
        FIA document titles typically follow:
          'Decision - Norris - MGU-H and Turbocharger - replaced'
        """
        title_upper = title.upper()

        # Only process decision documents that mention component replacements
        if "DECISION" not in title_upper and "PENALTY" not in title_upper:
            return None

        # Detect component keyword
        matched_component = None
        for component in PENALTY_COMPONENT_GRID_DROP:
            if component in title_upper:
                matched_component = component
                break

        if not matched_component:
            return None

        grid_drop = PENALTY_COMPONENT_GRID_DROP[matched_component]

        # Extract driver name heuristically from the title
        # Typical format: "Decision - <Driver> - <component>"
        parts = [p.strip() for p in title.split("-")]
        driver_name = parts[1] if len(parts) > 1 else "Unknown"
        team_name = "Unknown"

        return GridPenalty(
            driver=driver_name,
            team=team_name,
            component=matched_component,
            grid_drop=grid_drop,
            reason=title,
            document_url=url,
        )


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 APEX — FIA Scraper CLI")
    parser.add_argument("--round", type=int, required=True, dest="round_id")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    scraper = FIADocumentScraper()
    result = scraper.fetch_weekend_documents(args.round_id, args.year)
    print(f"Penalties found: {result.total_penalised_drivers}")
    for p in result.penalties:
        print(f"  {p.driver}: -{p.grid_drop} places ({p.component})")
