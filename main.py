"""AstrBot integration entry point for the NJU knowledge-gap report plugin."""

from __future__ import annotations

from astrbot.api import logger
from astrbot.api.star import Context, Star, register


@register(
    "astrbot_plugin_nju_qa_report",
    "whyself",
    "南京大学迎新问答采集与知识缺口日报（非官方）",
    "0.1.0",
)
class NjuQaReportPlugin(Star):
    """Plugin shell; domain services are implemented under :mod:`nju_report`."""

    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        logger.info("NJU QA report plugin scaffold loaded")

    async def terminate(self) -> None:
        """Release plugin resources during unload."""

        logger.info("NJU QA report plugin scaffold unloaded")
