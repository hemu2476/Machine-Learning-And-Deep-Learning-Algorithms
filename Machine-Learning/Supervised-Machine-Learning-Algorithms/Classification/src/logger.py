# ============================================================================
# Logging Module for K-NN Classification Pipeline
# ============================================================================
# Provides a factory function that produces a fully configured Python logger
# with both console and file handlers.  The file handler writes to the
# 'output/' directory alongside model results, keeping all artifacts together.
# ============================================================================

import logging
import os
import sys

from config import LoggingConfig, PathConfig


class LoggerFactory:
    """Factory responsible for creating and configuring pipeline loggers.

    Encapsulates all handler/formatter setup so that consumers simply call
    `LoggerFactory.create(...)` and receive a ready-to-use logger instance.

    Design decision -- Why a factory class instead of module-level init?
    A factory allows creating multiple independent loggers (e.g., for
    separate pipeline stages) while keeping configuration explicit.
    """

    # Detailed format with timestamps, module name, severity, and message
    _FORMAT = (
        "[%(asctime)s] [%(levelname)-8s] [%(name)-20s] %(message)s"
    )
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def create(
        cls,
        name: str,
        logging_config: LoggingConfig,
        path_config: PathConfig,
    ) -> logging.Logger:
        """Build and return a configured logger.

        Args:
            name:           Logger name (typically __name__ of the caller).
            logging_config: Logging settings (level, file toggle, filename).
            path_config:    Path settings (used to resolve log file location).

        Returns:
            A logging.Logger instance with console (and optionally file)
            handlers attached.
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, logging_config.log_level.upper(), logging.DEBUG))

        # Prevent duplicate handlers when the factory is called more than once
        # for the same logger name (e.g., during interactive development).
        if logger.handlers:
            return logger

        formatter = logging.Formatter(cls._FORMAT, datefmt=cls._DATE_FORMAT)

        # --- Console handler (always active) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger