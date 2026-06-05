import sys
from typing import Optional

from research_agent_system.utils.logger import get_logger

logger = get_logger(__name__)


class ResearchAgentException(Exception):

    def __init__(
        self,
        error_message: str,
        error_detail: Optional[sys] = None
    ):
        super().__init__(error_message)

        self.error_message = self._build_message(
            error_message,
            error_detail
        )

    def _build_message(
        self,
        error_message: str,
        error_detail
    ) -> str:

        if error_detail:

            _, _, exc_tb = error_detail.exc_info()

            if exc_tb:

                file_name = (
                    exc_tb.tb_frame.f_code.co_filename
                )

                line_number = exc_tb.tb_lineno

                msg = (
                    f"[{file_name}] "
                    f"line {line_number} -> "
                    f"{error_message}"
                )

                logger.error(msg)

                return msg

        logger.error(error_message)

        return error_message

    def __str__(self):

        return self.error_message