import json
import os
from datetime import datetime, timezone

from config import load_config


def _results_dir():
    results_dir = load_config()["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def current_settings():
    provider = load_config()["provider"]
    settings = {
        "model": provider["model"],
        "temperature": provider["temperature"],
        "max_tokens": provider["max_tokens"],
    }
    if "reasoning_effort" in provider:
        settings["reasoning_effort"] = provider["reasoning_effort"]
    if "reasoning_format" in provider:
        settings["reasoning_format"] = provider["reasoning_format"]
    return settings


def build_record(run_id, condition, prompt, raw_reply, parsed_words,
                  valid_words, invalid_words, score, status,
                  attempts=None, retries_used=0, reminders_used=0):
    return {
        "run_id": run_id,
        "condition": condition,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **current_settings(),
        "prompt": prompt,
        "attempts": attempts or [],
        "raw_reply": raw_reply,
        "parsed_words": parsed_words,
        "valid_words": valid_words,
        "invalid_words": invalid_words,
        "dat_score": score,
        "retries_used": retries_used,
        "reminders_used": reminders_used,
        "status": status,
    }


def save_run(record, filename):
    path = os.path.join(_results_dir(), filename)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path
