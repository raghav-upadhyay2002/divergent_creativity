import re
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

from .config import load_config

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

load_dotenv()

MAX_RATE_LIMIT_TRIES = 8
MAX_NETWORK_TRIES = 3
BACKOFF_BASE = 2
BACKOFF_CAP = 60
DAILY_LIMIT_SECONDS = 600

_client = None
_provider_config = None
_total_tokens = 0


class DailyLimitReached(Exception):
    """Raised when the API signals a daily (not just per-minute) limit."""

    def __init__(self, wait_seconds=None, message="Daily limit reached"):
        super().__init__(message)
        self.wait_seconds = wait_seconds


def _get_provider_config():
    """Load and cache the active config's provider settings."""
    global _provider_config
    if _provider_config is None:
        _provider_config = load_config()["provider"]
    return _provider_config


def _get_client():
    """Build and cache the OpenAI-compatible client for the active config."""
    global _client
    if _client is None:
        import os

        provider = _get_provider_config()
        api_key = os.environ[provider["api_key_env"]]
        _client = OpenAI(base_url=provider["base_url"], api_key=api_key)
    return _client


def _sleep(seconds):
    """Sleep for seconds (its own function so tests can patch it out)."""
    time.sleep(seconds)


def _extract_retry_after(error):
    """Read the Retry-After header (in seconds) off a rate-limit error, if present."""
    try:
        value = error.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _looks_like_daily_limit(error):
    """Guess whether a rate-limit error is a daily (not per-minute) limit."""
    return "daily" in str(error).lower()


def _backoff_seconds(tries):
    """Exponential backoff: 2, 4, 8, ... seconds, capped at BACKOFF_CAP."""
    return min(BACKOFF_BASE ** tries, BACKOFF_CAP)


def send(messages, max_tokens=None, client=None):
    """Send messages to the model with retry/backoff; return content, finish_reason, usage."""
    provider = _get_provider_config()
    active_client = client or _get_client()
    tokens_limit = max_tokens if max_tokens is not None else provider["max_tokens"]

    create_kwargs = {
        "model": provider["model"],
        "temperature": provider["temperature"],
        "max_tokens": tokens_limit,
    }
    if "reasoning_effort" in provider:
        create_kwargs["reasoning_effort"] = provider["reasoning_effort"]
    if "reasoning_format" in provider:
        # not a native SDK param -- passed straight through in the request body
        create_kwargs["extra_body"] = {"reasoning_format": provider["reasoning_format"]}

    rate_limit_tries = 0
    network_tries = 0

    while True:
        try:
            response = active_client.chat.completions.create(
                messages=messages,
                **create_kwargs,
            )
        except openai.RateLimitError as error:
            rate_limit_tries += 1
            retry_after = _extract_retry_after(error)

            if (retry_after is not None and retry_after > DAILY_LIMIT_SECONDS) or \
                    _looks_like_daily_limit(error):
                raise DailyLimitReached(wait_seconds=retry_after) from error

            if rate_limit_tries >= MAX_RATE_LIMIT_TRIES:
                raise

            wait = retry_after if retry_after is not None else _backoff_seconds(rate_limit_tries)
            _sleep(wait)
            continue
        except (openai.APIConnectionError, openai.InternalServerError):
            network_tries += 1
            if network_tries >= MAX_NETWORK_TRIES:
                raise
            _sleep(_backoff_seconds(network_tries))
            continue

        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            global _total_tokens
            _total_tokens += response.usage.total_tokens

        choice = response.choices[0]
        reply = choice.message.content or ""
        reply = _THINK_TAG_RE.sub("", reply).strip()

        return {
            "content": reply,
            "finish_reason": choice.finish_reason,
            "usage": usage,
        }


def total_tokens_used():
    """Return the running total of tokens used across all send() calls."""
    return _total_tokens


def print_token_usage():
    """Print the running total of tokens used so far."""
    print(f"Total tokens used: {_total_tokens}")
