import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

from config import load_config

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
    global _provider_config
    if _provider_config is None:
        _provider_config = load_config()["provider"]
    return _provider_config


def _get_client():
    global _client
    if _client is None:
        import os

        provider = _get_provider_config()
        api_key = os.environ[provider["api_key_env"]]
        _client = OpenAI(base_url=provider["base_url"], api_key=api_key)
    return _client


def _sleep(seconds):
    time.sleep(seconds)


def _extract_retry_after(error):
    try:
        value = error.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _looks_like_daily_limit(error):
    return "daily" in str(error).lower()


def _backoff_seconds(tries):
    return min(BACKOFF_BASE ** tries, BACKOFF_CAP)


def chat(messages, max_tokens=None, client=None):
    provider = _get_provider_config()
    active_client = client or _get_client()
    tokens_limit = max_tokens if max_tokens is not None else provider["max_tokens"]

    rate_limit_tries = 0
    network_tries = 0

    while True:
        try:
            response = active_client.chat.completions.create(
                model=provider["model"],
                messages=messages,
                temperature=provider["temperature"],
                reasoning_effort=provider["reasoning_effort"],
                max_tokens=tokens_limit,
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

        if response.usage is not None:
            global _total_tokens
            _total_tokens += response.usage.total_tokens

        reply = response.choices[0].message.content
        return reply or ""


def total_tokens_used():
    return _total_tokens


def print_token_usage():
    print(f"Total tokens used: {_total_tokens}")
