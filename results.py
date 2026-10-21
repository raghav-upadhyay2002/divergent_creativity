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
    return {
        "model": provider["model"],
        "temperature": provider["temperature"],
        "reasoning_effort": provider["reasoning_effort"],
        "max_tokens": provider["max_tokens"],
    }


def build_record(run_id, condition, prompt, raw_reply, parsed_words,
                  valid_words, invalid_words, score, status):
    return {
        "run_id": run_id,
        "condition": condition,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **current_settings(),
        "prompt": prompt,
        "raw_reply": raw_reply,
        "parsed_words": parsed_words,
        "valid_words": valid_words,
        "invalid_words": invalid_words,
        "dat_score": score,
        "status": status,
    }


def save_run(record, filename):
    path = os.path.join(_results_dir(), filename)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path
