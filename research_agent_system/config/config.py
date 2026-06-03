import os

from dataclasses import dataclass


@dataclass
class Config:

    # MODES
    dev_mode: bool = False

    # PIPELINE SETTINGS
    min_papers: int = 8
    topk: int = 5
    max_attempts: int = 3

    # STORAGE
    cache_dir: str = "cache"

    # MODELS
    primary_model: str = ("gemini-2.5-flash")

    # INIT
    def __post_init__(self):

        # DEV MODE
        if self.dev_mode:
            self.min_papers = 2
            self.topk = 2
            self.max_attempts = 1

        # DIRECTORIES

        os.makedirs(self.cache_dir,exist_ok=True)
        os.makedirs("data",exist_ok=True)
        os.makedirs("logs",exist_ok=True)