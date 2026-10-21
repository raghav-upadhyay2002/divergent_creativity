import json
import os
from datetime import datetime, timezone

from .config import load_config


def current_settings():
    """Return the active config's provider settings (model, temperature, etc.)."""
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


def build_record(run_id, condition, result):
    """Wrap a conversation run_solo()/run_pair() result into a saved record."""
    return {
        "run_id": run_id,
        "condition": condition,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "roles": result["roles"],
        "settings": current_settings(),
        "steps": result["steps"],
        "total_reminders_used": result["total_reminders_used"],
        "total_retries_used": result["total_retries_used"],
        "status": result["status"],
    }


def save_run(record, path):
    """Append one record as a line to a jsonl file, creating its folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def load_records(path):
    """Load every record from a jsonl file (an empty list if it doesn't exist)."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def partition_by_status(path, run_ids_of_interest):
    """Split a runs file into: ok run_ids, failed records to move out, lines to keep.

    Only records whose run_id is in run_ids_of_interest and whose status isn't
    "ok" are flagged to move -- everything else (including unrelated run_ids)
    is kept exactly as it was.
    """
    ok_ids = set()
    failed_records = []
    keep_lines = []

    if not os.path.exists(path):
        return ok_ids, failed_records, keep_lines

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            run_id = record.get("run_id")
            status = record.get("status")

            if run_id in run_ids_of_interest and status != "ok":
                failed_records.append(record)
            else:
                keep_lines.append(line)
                if status == "ok":
                    ok_ids.add(run_id)

    return ok_ids, failed_records, keep_lines


def move_failed_records(runs_path, failures_path, failed_records, keep_lines):
    """Append failed records to the failures file, then rewrite runs_path without them."""
    os.makedirs(os.path.dirname(failures_path), exist_ok=True)
    with open(failures_path, "a") as f:
        for record in failed_records:
            f.write(json.dumps(record) + "\n")

    with open(runs_path, "w") as f:
        for line in keep_lines:
            f.write(line + "\n")
