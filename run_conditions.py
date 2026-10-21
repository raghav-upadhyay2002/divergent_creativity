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


def _existing_run_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.add(json.loads(line)["run_id"])
    return ids


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
        "status": result["status"],
    }
    results.save_run(record, RESULTS_FILE)
    return record


def run_spec(spec):
    if spec["condition"] == "solo":
        return run_solo.run_once(spec["run_id"], filename=RESULTS_FILE)
    return run_pair_once(spec["run_id"], spec["condition"], spec["role_a"], spec["role_b"])


def build_shuffled_runs(config=None):
    config = config or load_config()
    experiment = config["experiment"]
    results_dir = config["paths"]["results_dir"]

    specs = _build_run_specs(
        experiment["conditions"], experiment["roles"], experiment["runs_per_condition"])

    rng = random.Random(experiment["random_seed"])
    rng.shuffle(specs)

    path = os.path.join(results_dir, RESULTS_FILE)
    done = _existing_run_ids(path)
    pending = [s for s in specs if s["run_id"] not in done]

    return specs, pending, done


def main():
    specs, pending, done = build_shuffled_runs()

    print(f"Total runs defined: {len(specs)}")
    print(f"Already completed (skipped): {len(done & {s['run_id'] for s in specs})}")
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
# Summary reporting (pairs-type conditions only; solo records have no steps)
# ---------------------------------------------------------------------------

def load_all_records(filename=RESULTS_FILE):
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


def summarize_condition(condition, records):
    copy_counts = {"round": [0, 0], "final": [0, 0]}  # [exact_copy count, total]

    start_scores = []
    final_scores = []
    final_shared_counts = []
    final_shared_words_union = set()

    for record in records:
        for s in record.get("steps", []):
            step_type = s.get("step_type")
            if step_type not in copy_counts:
                continue
            for label in ("agent_a", "agent_b"):
                agent_record = s[label]
                if agent_record and agent_record["comparison"] is not None:
                    copy_counts[step_type][1] += 1
                    if agent_record["comparison"]["exact_copy"]:
                        copy_counts[step_type][0] += 1

        start = _step(record, "start")
        if start and start["team_score"] is not None:
            start_scores.append(start["team_score"])

        final = _step(record, "final")
        if final and final["team_score"] is not None:
            final_scores.append(final["team_score"])
        if final:
            final_shared_counts.append(len(final["shared_words"]))
            final_shared_words_union.update(final["shared_words"])

    def avg(values):
        return sum(values) / len(values) if values else None

    return {
        "condition": condition,
        "n_runs": len(records),
        "exact_copy_rounds": tuple(copy_counts["round"]),
        "exact_copy_final": tuple(copy_counts["final"]),
        "avg_start_score": avg(start_scores),
        "avg_final_score": avg(final_scores),
        "avg_final_shared_words": avg(final_shared_counts),
        "final_shared_words_seen": sorted(final_shared_words_union),
    }


def print_summary(records):
    by_condition = {}
    for r in records:
        by_condition.setdefault(r["condition"], []).append(r)

    # solo has no steps/rounds, so it's excluded from this (pairs-only) summary
    pair_conditions = [c for c in by_condition if c != "solo"]
    summaries = [summarize_condition(c, by_condition[c]) for c in pair_conditions]

    def fmt_ratio(pair):
        count, total = pair
        if total == 0:
            return "n/a"
        return f"{count}/{total} ({100 * count / total:.0f}%)"

    headers = ("condition", "n", "exact copy (rounds)", "exact copy (final)",
               "avg start score", "avg final score", "avg shared @final")
    rows = []
    for s in summaries:
        rows.append((
            s["condition"], s["n_runs"],
            fmt_ratio(s["exact_copy_rounds"]), fmt_ratio(s["exact_copy_final"]),
            f"{s['avg_start_score']:.2f}" if s["avg_start_score"] is not None else "-",
            f"{s['avg_final_score']:.2f}" if s["avg_final_score"] is not None else "-",
            f"{s['avg_final_shared_words']:.1f}" if s["avg_final_shared_words"] is not None else "-",
        ))

    if not rows:
        print("No pairs-type records to summarize yet.")
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


if __name__ == "__main__":
    main()
