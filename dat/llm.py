import json
import os
import re
import subprocess
import time
from pathlib import Path

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

_REPO_ROOT = Path(__file__).parent.parent

# Flags that keep the Claude Code CLI from loading MCP servers, skills, plugins,
# hooks, CLAUDE.md and settings files (measured: ~850 tokens/call vs ~69,000 without).
_CLAUDE_ISOLATION_FLAGS = ["--safe-mode", "--strict-mcp-config", "--disable-slash-commands"]
_CLAUDE_ENV_REMOVED = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
_CLAUDE_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "overloaded", "529")
_CLAUDE_DAILY_LIMIT_MARKERS = ("daily", "usage limit", "limit reached")

_client = None
_provider_config = None
_total_tokens = 0
_claude_token = None
_claude_version = None


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


class ClaudeCLIError(RuntimeError):
    """Raised when the Claude Code CLI keeps failing after retries (token already scrubbed)."""


class ClaudeTokenError(RuntimeError):
    """Raised when the Claude token file is missing or empty (never falls back to a login)."""


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


def send(messages, max_tokens=None, client=None, session_id=None):
    """Send messages to the model with retry/backoff; return content, finish_reason, usage.

    session_id is only used by the claude_cli provider (it resumes that session and sends
    just the latest message); the Groq provider ignores it and sends the whole history.
    """
    provider = _get_provider_config()
    if provider.get("type") == "claude_cli":
        return _send_claude_cli(messages, session_id)
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


# ---------------------------------------------------------------------------
# claude_cli provider: runs the Claude Code CLI as a subprocess
# ---------------------------------------------------------------------------


def _resolve_path(path):
    """Resolve a config path relative to the repo root (absolute paths are kept)."""
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _load_claude_token():
    """Read (and cache) the OAuth token from the configured file; never fall back."""
    global _claude_token
    if _claude_token is None:
        token_path = _resolve_path(_get_provider_config()["token_file"])
        try:
            token = token_path.read_text().strip()
        except OSError as error:
            raise ClaudeTokenError(f"Claude token file not readable: {token_path}") from error
        if not token:
            raise ClaudeTokenError(f"Claude token file is empty: {token_path}")
        _claude_token = token
    return _claude_token


def _scrub(text, token):
    """Replace any occurrence of the token in text with a placeholder."""
    text = text or ""
    return text.replace(token, "[REDACTED]") if token else text


def _claude_env(token):
    """Environment for the subprocess: token set, API-key style auth removed."""
    env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_REMOVED}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def _claude_run_dir():
    """The one fixed (empty, gitignored) folder every call runs from, so --resume works."""
    run_dir = _resolve_path(_get_provider_config().get("run_dir", ".claude_runs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def claude_code_version():
    """Return `claude --version` (checked once, then cached). Needs no token."""
    global _claude_version
    if _claude_version is None:
        binary = _get_provider_config().get("claude_bin", "claude")
        env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_REMOVED}
        completed = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                    env=env, stdin=subprocess.DEVNULL, timeout=60)
        _claude_version = completed.stdout.strip()
    return _claude_version


def startup_checks():
    """Fail early for the claude_cli provider: token present, version recorded."""
    if _get_provider_config().get("type") == "claude_cli":
        _load_claude_token()
        claude_code_version()


def _claude_command(provider, prompt, session_id):
    """Build the argv. Every flag is passed on every call, including --resume calls."""
    command = [
        provider.get("claude_bin", "claude"), "-p", prompt,
        "--model", provider["model"],
        "--effort", provider["effort"],
        "--system-prompt", provider["system_prompt"],
        "--tools", "",
        "--output-format", "json",
        *_CLAUDE_ISOLATION_FLAGS,
    ]
    if session_id is not None:
        command += ["--resume", session_id]
    return command


def _claude_attempt(provider, prompt, session_id, token):
    """Run the CLI once. Return (parsed_json, None) or (None, (kind, scrubbed_message))."""
    try:
        completed = subprocess.run(
            _claude_command(provider, prompt, session_id),
            capture_output=True, text=True, env=_claude_env(token), cwd=_claude_run_dir(),
            stdin=subprocess.DEVNULL, timeout=provider.get("timeout_seconds", 300),
        )
    except subprocess.TimeoutExpired:
        return None, ("network", "claude CLI timed out")
    except OSError as error:
        return None, ("network", _scrub(f"could not run claude CLI: {error}", token))

    try:
        data = json.loads(completed.stdout)
    except ValueError:
        detail = _scrub((completed.stderr or completed.stdout or "")[-300:], token)
        return None, ("network", f"claude CLI exit {completed.returncode}, bad JSON: {detail}")

    if data.get("is_error") or completed.returncode != 0:
        detail = _scrub(str(data.get("result") or completed.stderr or "")[-300:], token)
        lowered = detail.lower()
        if any(marker in lowered for marker in _CLAUDE_DAILY_LIMIT_MARKERS):
            return None, ("daily", detail)
        if any(marker in lowered for marker in _CLAUDE_RATE_LIMIT_MARKERS):
            return None, ("rate_limit", detail)
        return None, ("network", detail or f"claude CLI exit {completed.returncode}")

    return data, None


def _send_claude_cli(messages, session_id):
    """claude_cli equivalent of send(): same retry/backoff/failure handling as Groq.

    Note: a failed call that used --resume still writes the user message into the
    session, so a retry leaves that message in it twice. We can't undo that, so the
    number of such failures is returned as failed_calls_on_resumed_session.
    """
    global _total_tokens
    provider = _get_provider_config()
    token = _load_claude_token()
    version = claude_code_version()
    prompt = messages[-1]["content"]

    rate_limit_tries = 0
    network_tries = 0
    failures = []
    failed_on_resume = 0

    while True:
        data, failure = _claude_attempt(provider, prompt, session_id, token)
        if failure is None:
            break

        kind, detail = failure
        failures.append(detail)
        if session_id is not None:
            failed_on_resume += 1

        if kind == "daily":
            raise DailyLimitReached(message=detail)
        if kind == "rate_limit":
            rate_limit_tries += 1
            if rate_limit_tries >= MAX_RATE_LIMIT_TRIES:
                raise ClaudeCLIError(detail)
            _sleep(_backoff_seconds(rate_limit_tries))
        else:
            network_tries += 1
            if network_tries >= MAX_NETWORK_TRIES:
                raise ClaudeCLIError(detail)
            _sleep(_backoff_seconds(network_tries))

    raw_usage = data.get("usage") or {}
    in_tokens = raw_usage.get("input_tokens", 0)
    cache_created = raw_usage.get("cache_creation_input_tokens", 0)
    cache_read = raw_usage.get("cache_read_input_tokens", 0)
    out_tokens = raw_usage.get("output_tokens", 0)
    prompt_tokens = in_tokens + cache_created + cache_read
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": prompt_tokens + out_tokens,
        "input_tokens": in_tokens,
        "cache_creation_input_tokens": cache_created,
        "cache_read_input_tokens": cache_read,
    }
    _total_tokens += usage["total_tokens"]

    model_usage = data.get("modelUsage") or {}
    return {
        "content": (data.get("result") or "").strip(),
        "finish_reason": data.get("stop_reason"),
        "usage": usage,
        "session_id": data.get("session_id"),
        "model": next(iter(model_usage), None),
        "effort": provider["effort"],
        "temperature": None,
        "claude_code_version": version,
        "failed_calls": failures,
        "failed_calls_on_resumed_session": failed_on_resume,
    }


def total_tokens_used():
    """Return the running total of tokens used across all send() calls."""
    return _total_tokens


def print_token_usage():
    """Print the running total of tokens used so far."""
    print(f"Total tokens used: {_total_tokens}")
