import re
import time
import random

from google import genai as google_genai
from groq import Groq

from research_agent_system.config.settings import (
    GOOGLE_API_KEY,
    GROQ_API_KEY
)

from research_agent_system.utils.logger import (
    get_logger
)


logger = get_logger(__name__)

# CLIENTS
_gemini_client = google_genai.Client(api_key=GOOGLE_API_KEY)
_groq_client = Groq(api_key=GROQ_API_KEY.strip())

# GROQ FALLBACK
def call_groq(
    prompt: str,
    max_tokens: int = 2048
) -> str:

    try:

        response = (
            _groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.2,

                max_tokens=max_tokens,
            )
        )

        result = (
            response.choices[0]
            .message.content
        )

        return result.strip()

    except Exception as e:

        logger.error(
            f"Groq error: {e}"
        )

        return ""


# PRIMARY LLM CALL
def call_llm(
    prompt: str,
    retries: int = 2
) -> str:

    for i in range(retries):

        try:

            response = (
                _gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
            )

            result = response.text

            if not result:

                raise ValueError(
                    "Empty Gemini response"
                )

            result = result.strip()

            return result

        except Exception as e:

            err = str(e)

            logger.warning(
                f"Gemini retry "
                f"{i+1}: {err[:120]}"
            )

            # RATE LIMIT
            if (
                "429" in err or
                "RESOURCE_EXHAUSTED" in err
            ):

                match = re.search(
                    r"retryDelay.*?(\\d+)s",
                    err
                )

                wait = (
                    int(match.group(1)) + 2
                    if match
                    else 60
                )

                logger.info(
                    f"Rate limited "
                    f"-> waiting {wait}s"
                )

                time.sleep(wait)

                break

            # NORMAL RETRY
            if i < retries - 1:
                time.sleep((2 ** i)+ random.uniform(1, 3))

    # GROQ FALLBACK
    logger.info("Switching to Groq fallback")

    result = call_groq(prompt)

    if result:

        logger.info("Groq fallback success")
        return result

    logger.error("Both models failed")

    return ""