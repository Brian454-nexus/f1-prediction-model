"""
F1 APEX — Centralised Logger
Provides a beautifully formatted structured logging format across all modules.
"""

import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Custom rich theme to match our Pit Wall neon aesthetic!
f1_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "danger": "bold red",
    "f1": "bold magenta",
})

console = Console(theme=f1_theme)

# We only want the file handler to keep the raw text format since we don't want rich ANSI codes in text files
FILE_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger with a consistent format.
    Terminal logs are rendered via rich for a gorgeous F1 CLI experience.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Stdout Rich handler ─────────────────────────────────────────────────
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False, # cleaner console
    )
    rich_handler.setLevel(level)
    # The RichHandler automatically nicely formats the message. We don't want standard format interpolating here.
    logger.addHandler(rich_handler)

    # ── File handler ────────────────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "f1_apex.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
