"""Non-blocking bridge from AstrBot's event loop to SQLite capture writes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

from .message_capture import MessageCaptureService
from .models import MessageEnvelope


class AsyncCaptureWriter:
    """Serialize SQLite capture writes outside AstrBot's event-loop thread."""

    def __init__(
        self,
        service: MessageCaptureService,
        *,
        max_queue_size: int = 5000,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size 必须大于 0")
        self._service = service
        self._queue: asyncio.Queue[MessageEnvelope] = asyncio.Queue(maxsize=max_queue_size)
        self._on_error = on_error
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nju-report-writer",
        )
        self._executor_closed = False
        self._task: asyncio.Task[None] | None = None
        self._accepting = False
        self.submitted_count = 0
        self.dropped_count = 0
        self.write_error_count = 0

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start one writer task on the current running event loop."""

        if self.running:
            return
        if self._executor_closed:
            raise RuntimeError("capture writer 已关闭，不能重新启动")
        self._accepting = True
        self._task = asyncio.create_task(
            self._run(),
            name="nju-report-message-writer",
        )

    def submit(self, message: MessageEnvelope) -> bool:
        """Enqueue immediately; never wait inside the AstrBot message handler."""

        if not self._accepting or not self.running:
            self.dropped_count += 1
            return False
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            self.dropped_count += 1
            return False
        self.submitted_count += 1
        return True

    async def flush(self, *, timeout_seconds: float = 30.0) -> None:
        """Wait until all currently queued messages finish writing."""

        await asyncio.wait_for(self._queue.join(), timeout=timeout_seconds)

    async def close(self, *, drain_timeout_seconds: float = 30.0) -> None:
        """Stop accepting messages, drain when possible, and stop the worker."""

        self._accepting = False
        task = self._task
        if task is None:
            await self._shutdown_executor()
            return
        try:
            await self.flush(timeout_seconds=drain_timeout_seconds)
        except TimeoutError:
            self._discard_pending()
            # The only remaining item may still be running in the dedicated
            # executor; executor shutdown below waits for it before DB close.
            with suppress(TimeoutError):
                await self.flush(timeout_seconds=10.0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._task = None
        await self._shutdown_executor()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            message = await self._queue.get()
            try:
                await loop.run_in_executor(
                    self._executor,
                    self._service.capture,
                    message,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.write_error_count += 1
                if self._on_error is not None:
                    # Error reporting must not terminate the only writer.
                    with suppress(Exception):
                        self._on_error(exc)
            finally:
                self._queue.task_done()

    def _discard_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self.dropped_count += 1
                self._queue.task_done()

    async def _shutdown_executor(self) -> None:
        if self._executor_closed:
            return
        self._executor_closed = True
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
