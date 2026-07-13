from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import TypeVar

from legopolitics.pipeline.context import PipelineContext
from legopolitics.schemas import ErrorRecord
from legopolitics.utils.memory import memory_snapshot

T = TypeVar("T")


def run_stage(
    context: PipelineContext,
    video_id: str,
    stage: str,
    cache_key: str,
    function: Callable[[], T],
    recoverable: bool = False,
) -> T | None:
    started = context.manifest.begin_stage(context.run_id, video_id, stage, cache_key)
    context.logger.info("Starting stage: %s", stage)
    try:
        result = function()
        context.manifest.finish_stage(
            context.run_id, video_id, stage, started, metadata={"memory": memory_snapshot()}
        )
        context.logger.info("Completed stage: %s", stage)
        return result
    except Exception as exc:
        trace = traceback.format_exc()
        error_id = context.manifest.record_error(
            context.run_id,
            video_id,
            stage,
            None,
            exc,
            trace,
            recoverable=recoverable,
            metadata={"memory": memory_snapshot()},
        )
        context.errors.append(
            ErrorRecord(
                error_id=error_id,
                video_id=video_id,
                stage=stage,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback=trace,
                recoverable=recoverable,
                memory_state=memory_snapshot(),
            )
        )
        context.manifest.finish_stage(
            context.run_id, video_id, stage, started, status="failed", error_message=str(exc)
        )
        context.logger.exception("Stage failed: %s", stage)
        if recoverable:
            return None
        raise
