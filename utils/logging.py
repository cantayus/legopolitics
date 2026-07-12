from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    console: bool = True,
    json_log: bool = True,
) -> logging.Logger:
    logger = logging.getLogger("legopolitics")
    logger.setLevel(level.upper())
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        text_handler = RotatingFileHandler(
            log_dir / "run.log", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
        )
        text_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(text_handler)
        if json_log:
            json_handler = RotatingFileHandler(
                log_dir / "run.jsonl", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
            )
            json_handler.setFormatter(JsonFormatter())
            logger.addHandler(json_handler)
    return logger
