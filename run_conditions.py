import argparse
import json
import os
import random
from datetime import datetime, timezone

import conversation
import llm
import results
import run_solo
from config import load_config

RESULTS_FILE = "pairs.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the mixed-condition DAT experiment.")
    parser.add_argument("--config", default="config.json",
                         help="Path to the config file to use (default: config.json)")
    parser.add_argument("--out", default="pairs.jsonl",
                         help="Results filename inside paths.results_dir (default: pairs.jsonl)")
    return parser.parse_args()


def _build_run_specs(conditions, roles, runs_per_condition):
    """Build one run spec per (condition, run index). same_role alternates
    between roles[0]+roles[0] and roles[1]+roles[1] across its runs."""
    specs = []
    for condition in conditions:
        for i in range(runs_per_condition):
            run_id = f"{condition}-{i:03d}"

            if condition == "no_roles":
                role_a, role_b = None, None
            elif condition == "different_roles":
                role_a, role_b = roles[0], roles[1]
            elif condition == "same_role":
                role = roles[i % len(roles)]
                role_a, role_b = role, role
            elif condition == "solo":
                role_a, role_b = None, None
            else:
                raise ValueError(f"Unknown condition: {condition}")

            specs.append({
                "condition": condition,
                "run_id": run_id,
                "role_a": role_a,
                "role_b": role_b,
            })
    return specs


def _failures_filename(filename):
    """pairs_qwen.jsonl -> failures_qwen.jsonl; pairs.jsonl -> failures.jsonl;
    anything else -> failures_<filename>."""
    if filename.startswith("pairs_"):
        return "failures_" + filename[len("pairs_"):]
    if filename == "pairs.jsonl":
        return "failures.jsonl"
    return "failures_" + filename


def _partition_existing_records(path, run_ids_of_interest):
    """Split existing records into: ok run_ids (to skip), failed records for run_ids
    we're about to retry (to move out to the failures file), and lines to keep as-is."""
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


def run_pair_once(run_id, condition, role_a, role_b):
    result = conversation.run_pair(role_a, role_b)
    record = {
        "run_id": run_id,
        "condition": condition,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "roles": result["roles"],
        "settings": results.current_settings(),
        "steps": result["steps"],
        "total_reminders_used": result["total_reminders_used"],
        "total_retries_used": result["total_retries_used"],
        "status": result["status"],
    }
    results.save_run(record, RESULTS_FILE)
    return record


def run_spec(spec):
    if spec["condition"] == "solo":
        return run_solo.run_once(spec["run_id"], filename=RESULTS_FILE)
    return run_pair_once(spec["run_id"], spec["condition"], spec["role_a"], spec["role_b"])


def build_shuffled_runs(config=None):
    """Build the full shuffled run list. Only run_ids already saved with status
    "ok" are skipped; anything else (including runs never attempted) is pending.
    Any existing non-"ok" record for a run we're about to retry is moved out to
    the failures file first, so it's kept rather than overwritten or lost."""
    config = config or load_config()
    experiment = config["experiment"]
    results_dir = config["paths"]["results_dir"]

    specs = _build_run_specs(
        experiment["conditions"], experiment["roles"], experiment["runs_per_condition"])

    rng = random.Random(experiment["random_seed"])
    rng.shuffle(specs)

    run_ids_of_interest = {s["run_id"] for s in specs}
    path = os.path.join(results_dir, RESULTS_FILE)

    ok_ids, failed_records, keep_lines = _partition_existing_records(path, run_ids_of_interest)

    if failed_records:
        failures_path = os.path.join(results_dir, _failures_filename(RESULTS_FILE))
        with open(failures_path, "a") as f:
            for record in failed_records:
                f.write(json.dumps(record) + "\n")
        with open(path, "w") as f:
            for line in keep_lines:
                f.write(line + "\n")

    pending = [s for s in specs if s["run_id"] not in ok_ids]

    return specs, pending, ok_ids, len(failed_records)


def main():
    global RESULTS_FILE

    args = parse_args()
    os.environ["DAT_CONFIG_PATH"] = args.config
    RESULTS_FILE = args.out

    print(f"Using config: {args.config}")
    print(f"Using results file: {RESULTS_FILE}")

    specs, pending, done, n_moved = build_shuffled_runs()

    if n_moved:
        print(f"Moved {n_moved} old failed record(s) to "
              f"{_failures_filename(RESULTS_FILE)} (kept, not deleted) -- will retry them")

    print(f"Total runs defined: {len(specs)}")
    print(f"Already completed (skipped, status=ok): {len(done & {s['run_id'] for s in specs})}")
    print(f"Pending this call: {len(pending)}")

    completed = []
    for spec in pending:
        try:
            record = run_spec(spec)
        except llm.DailyLimitReached:
            print(f"Daily limit reached. Completed {len(completed)} new run(s) before stopping.")
            break
        completed.append(record)
        print(f"  done: {spec['run_id']} ({spec['condition']}) -> status={record['status']}")

    print_summary(load_all_records())
    return completed


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def load_all_records(filename=None):
    filename = filename if filename is not None else RESULTS_FILE
    config = load_config()
    path = os.path.join(config["paths"]["results_dir"], filename)
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _step(record, step_name):
    return next((s for s in record.get("steps", []) if s["step"] == step_name), None)


def _avg(values):
    return sum(values) / len(values) if values else None


def summarize_solo(records):
    ok_records = [r for r in records if r["status"] == "ok"]
    scores = [r["dat_score"] for r in ok_records]
    return {
        "condition": "solo",
        "n_runs": len(records),
        "n_failed": len(records) - len(ok_records),
        "avg_score": _avg(scores),
    }


def summarize_condition(condition, records):
    copy_counts = {"round": [0, 0], "final": [0, 0]}  # [exact_copy count, total]

    start_scores = []
    final_scores = []
    start_all10_scores = []
    final_all10_scores = []
    words_changed_rounds = []
    final_shared_counts = []
    final_shared_words_union = set()

    for record in records:
        for s in record.get("steps", []):
            step_type = s.get("step_type")

            if step_type in copy_counts:
                for label in ("agent_a", "agent_b"):
                    agent_record = s[label]
                    comparison = agent_record.get("comparison") if agent_record else None
                    if comparison is not None:
                        copy_counts[step_type][1] += 1
                        if comparison["exact_copy"]:
                            copy_counts[step_type][0] += 1

            if step_type == "round":
                for label in ("agent_a", "agent_b"):
                    agent_record = s[label]
                    changed = agent_record.get("words_changed_since_previous") if agent_record else None
                    if changed is not None:
                        words_changed_rounds.append(changed)

        start = _step(record, "start")
        if start and start["team_score"] is not None:
            start_scores.append(start["team_score"])
        if start and start["agent_a"] and start["agent_b"]:
            a10 = start["agent_a"].get("score_all10")
            b10 = start["agent_b"].get("score_all10")
            if a10 is not None and b10 is not None:
                start_all10_scores.append((a10 + b10) / 2)

        final = _step(record, "final")
        if final and final["team_score"] is not None:
            final_scores.append(final["team_score"])
        if final and final["agent_a"] and final["agent_b"]:
            a10 = final["agent_a"].get("score_all10")
            b10 = final["agent_b"].get("score_all10")
            if a10 is not None and b10 is not None:
                final_all10_scores.append((a10 + b10) / 2)
        # only count shared words for a final step that actually succeeded --
        # a failed final step can still produce a (degenerate) shared_words value
        if final and final["status"] == "ok":
            final_shared_counts.append(len(final["shared_words"]))
            final_shared_words_union.update(final["shared_words"])

    n_failed = len([r for r in records if r["status"] != "ok"])

    return {
        "condition": condition,
        "n_runs": len(records),
        "n_failed": n_failed,
        "exact_copy_rounds": tuple(copy_counts["round"]),
        "exact_copy_final": tuple(copy_counts["final"]),
        "avg_start_score": _avg(start_scores),
        "avg_final_score": _avg(final_scores),
        "avg_start_score_all10": _avg(start_all10_scores),
        "avg_final_score_all10": _avg(final_all10_scores),
        "avg_words_changed_rounds": _avg(words_changed_rounds),
        "avg_final_shared_words": _avg(final_shared_counts),
        "final_shared_words_seen": sorted(final_shared_words_union),
    }


def print_summary(records):
    by_condition = {}
    for r in records:
        by_condition.setdefault(r["condition"], []).append(r)

    pair_conditions = [c for c in by_condition if c != "solo"]
    summaries = [summarize_condition(c, by_condition[c]) for c in pair_conditions]
    solo_summary = summarize_solo(by_condition["solo"]) if "solo" in by_condition else None

    def fmt_ratio(pair):
        count, total = pair
        if total == 0:
            return "n/a"
        return f"{count}/{total} ({100 * count / total:.0f}%)"

    def fmt(value, decimals=2):
        return f"{value:.{decimals}f}" if value is not None else "-"

    headers = ("condition", "n", "failed", "avg score (solo)", "exact copy (rounds)",
               "exact copy (final)", "avg start score", "avg final score",
               "avg score_all10 (start)", "avg score_all10 (final)",
               "avg words changed (rounds)", "avg shared @final")
    rows = []

    if solo_summary:
        rows.append((
            solo_summary["condition"], solo_summary["n_runs"], solo_summary["n_failed"],
            fmt(solo_summary["avg_score"]),
            "-", "-", "-", "-", "-", "-", "-", "-",
        ))

    for s in summaries:
        rows.append((
            s["condition"], s["n_runs"], s["n_failed"], "-",
            fmt_ratio(s["exact_copy_rounds"]), fmt_ratio(s["exact_copy_final"]),
            fmt(s["avg_start_score"]), fmt(s["avg_final_score"]),
            fmt(s["avg_start_score_all10"]), fmt(s["avg_final_score_all10"]),
            fmt(s["avg_words_changed_rounds"], 1), fmt(s["avg_final_shared_words"], 1),
        ))

    if not rows:
        print("No records to summarize yet.")
        return

    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]

    def print_row(row):
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))

    print_row(headers)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print_row(row)

    print("\nShared words seen at the final step, by condition:")
    for s in summaries:
        words = ", ".join(s["final_shared_words_seen"]) if s["final_shared_words_seen"] else "(none)"
        print(f"  {s['condition']}: {words}")

    if "same_role" in by_condition:
        print("\nsame_role runs -- roles used (confirming PM+PM / programmer+programmer alternation):")
        for r in sorted(by_condition["same_role"], key=lambda r: r["run_id"]):
            roles = r["roles"]
            print(f"  {r['run_id']}: A={roles['A']!r}, B={roles['B']!r}")


if __name__ == "__main__":
    main()
