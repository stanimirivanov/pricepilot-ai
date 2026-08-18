# src/pricepilot/utils/logging_config.py
import sys

from loguru import logger

from pricepilot.config.settings import Settings


def setup_logging(settings: Settings) -> None:
    """Configure logging"""

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>",
    )

    # File handler
    logger.add(
        "logs/pricepilot_{time:YYYY-MM-DD}.log",
        rotation="500 MB",
        retention="10 days",
        level="DEBUG",
    )
