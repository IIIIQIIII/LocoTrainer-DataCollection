"""LocoTrainer — lightweight code agent for codebase exploration and analysis."""

__version__ = "0.1.1"

from .agent import Agent
from .config import Config
from .tools import ToolExecutor
from .repo import ensure_ms_swift_repo, ensure_repo

__all__ = ["Agent", "Config", "ToolExecutor", "ensure_ms_swift_repo", "ensure_repo", "__version__"]
