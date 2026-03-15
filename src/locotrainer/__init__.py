"""LocoTrainer — lightweight code agent for codebase exploration and analysis."""

__version__ = "0.1.1"

from .agent import Agent
from .config import Config
from .tools import ToolExecutor

__all__ = ["Agent", "Config", "ToolExecutor", "__version__"]
