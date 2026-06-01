# research_agent_system/agents/base_agent.py

import sys
from abc import ABC, abstractmethod
from research_agent_system.config.config import Config
from research_agent_system.utils.logger import get_logger
from research_agent_system.utils.exception import ResearchAgentException

class BaseAgent(ABC):
    """
    Base class for all agents.
    Every agent gets: cfg, logger, safe_run wrapper.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Each agent implements its own run()."""
        ...

    def safe_run(self, *args, **kwargs):
        """
        Wraps run() with exception handling.
        Use this from pipeline instead of run() directly.
        """
        try:
            return self.run(*args, **kwargs)
        except Exception as e:
            raise ResearchAgentException(str(e), sys) from e