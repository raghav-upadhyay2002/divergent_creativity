import argparse
import os
import statistics
from collections import Counter

from dat import scoring
from dat.storage import load_records


def parse_args():
    """Parse the results folder to summarize, e.g. results/qwen."""
    parser = argparse.ArgumentParser(description="Print summary tables for a results folder.")
    parser.add_argument("results_dir", help="e.g. results/qwen")
    return parser.parse_args()


def _avg(values):
    """Return the mean of values, or None if there are none."""
    return sum(values) / len(values) if values else None


def _step(record, step_name):
    """Find one named step in a pair run's steps list."""
    return next((s for s in record.get("steps", []) if s["step"] == step_name), None)


def _is_flat_solo(record):
    """True for the old flat solo schema that predates the "steps" list."""
    return "steps" not in record


def _solo_agent(record):
    """Return a solo run's single agent data, whichever schema it's stored in."""
    if _is_flat_solo(record):
        return {"dat_score": record["dat_score"], "parsed_words": record["parsed_words"]}
    return record["steps"][0]["agent_a"]


def all_valid_words(parsed_words):
    """Validate up to 10 words from a reply, like Wang/Olson's get_noun_count."""
    _score, used, _dropped = scoring.score_details_all10(parsed_words)
    return used


def _count_by_source(records):
    """Split solo runs into "solo baseline" vs "conditions test" by run_id.

    The old standalone solo script numbered its own runs 000..019 and never
    repeated an id within itself, so a run_id that shows up more than once
    means one copy came from the separate mixed-conditions run (which always
    contributes a small, fixed number of solo slots). Everything else is the
    baseline. This only applies to this historical data -- solo is just one
    condition in the current run_experiment.py, with no such split to make.
    """
    run_id_counts = Counter(r["run_id"] for r in records)
    n_conditions_test = len([run_id for run_id, count in run_id_counts.items() if count > 1])
    n_baseline = len(records) - n_conditions_test
    return n_baseline, n_conditions_test


def summarize_solo(records):
    """Build the solo condition's summary: score stats and word diversity."""
    ok_records = [r for r in records if r["status"] == "ok"]
    agents = [_solo_agent(r) for r in ok_records]
    scores = [a["dat_score"] for a in agents]

    all_words = []
    for a in agents:
        all_words.extend(all_valid_words(a["parsed_words"]))

    n_baseline, n_conditions_test = _count_by_source(records)

    return {
        "n_runs": len(records),
        "n_baseline": n_baseline,
        "n_conditions_test": n_conditions_test,
        "n_failed": len(records) - len(ok_records),
        "avg_score": _avg(scores),
        "sd_score": statistics.stdev(scores) if len(scores) > 1 else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "unique_words": len(set(all_words)),
        "top_words": Counter(all_words).most_common(10),
    }


def summarize_condition(condition, records):
    """Build one pairs condition's summary: scores, copying, and shared words."""
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
        # a failed final step can still produce a degenerate shared_words value
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


def print_pairs_table(summaries):
    """Print the comparison table for the non-solo conditions."""
    def fmt_ratio(pair):
        count, total = pair
        if total == 0:
            return "n/a"
        return f"{count}/{total} ({100 * count / total:.0f}%)"

    def fmt(value, decimals=2):
        return f"{value:.{decimals}f}" if value is not None else "-"

    headers = ("condition", "n", "failed", "exact copy (rounds)", "exact copy (final)",
               "avg start score", "avg final score", "avg score_all10 (start)",
               "avg score_all10 (final)", "avg words changed (rounds)", "avg shared @final")
    rows = [(
        s["condition"], s["n_runs"], s["n_failed"],
        fmt_ratio(s["exact_copy_rounds"]), fmt_ratio(s["exact_copy_final"]),
        fmt(s["avg_start_score"]), fmt(s["avg_final_score"]),
        fmt(s["avg_start_score_all10"]), fmt(s["avg_final_score_all10"]),
        fmt(s["avg_words_changed_rounds"], 1), fmt(s["avg_final_shared_words"], 1),
    ) for s in summaries]

    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]

    def print_row(row):
        # left-justify the condition label, right-justify every data column so
        # numbers line up on the right instead of trailing off raggedly
        cells = [str(row[0]).ljust(widths[0])]
        cells += [str(v).rjust(widths[i]) for i, v in enumerate(row[1:], start=1)]
        print(" | ".join(cells))

    print_row(headers)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print_row(row)

    print("\nShared words seen at the final step, by condition:")
    for s in summaries:
        words = ", ".join(s["final_shared_words_seen"]) if s["final_shared_words_seen"] else "(none)"
        print(f"  {s['condition']}: {words}")


def print_solo_summary(summary):
    """Print the solo condition's score stats and word diversity."""
    print(f"Total runs: {summary['n_runs']} "
          f"({summary['n_baseline']} from the solo baseline, "
          f"{summary['n_conditions_test']} from the conditions test)  "
          f"(failed: {summary['n_failed']})")
    print(f"Average score: {summary['avg_score']:.2f}" if summary["avg_score"] is not None
          else "No successful runs to summarize.")
    if summary["sd_score"] is not None:
        print(f"SD: {summary['sd_score']:.2f}")
        print(f"Min: {summary['min_score']:.2f}")
        print(f"Max: {summary['max_score']:.2f}")

    print(f"\nUnique words across all lists (up to 10 valid words each): {summary['unique_words']}")
    print("\nTop 10 most common words:")
    for word, count in summary["top_words"]:
        print(f"  {word}: {count}")


def print_same_role_roles(records):
    """Print each same_role run's actual roles, to confirm the PM/programmer alternation."""
    same_role_records = [r for r in records if r["condition"] == "same_role"]
    if not same_role_records:
        return
    print("\nsame_role runs -- roles used:")
    for r in sorted(same_role_records, key=lambda r: r["run_id"]):
        roles = r["roles"]
        print(f"  {r['run_id']}: A={roles['A']!r}, B={roles['B']!r}")


def main():
    """Load a results folder's runs and print its summary tables."""
    args = parse_args()
    records = load_records(os.path.join(args.results_dir, "runs.jsonl"))

    by_condition = {}
    for r in records:
        by_condition.setdefault(r["condition"], []).append(r)

    if "solo" in by_condition:
        print("=== solo ===")
        print_solo_summary(summarize_solo(by_condition["solo"]))
        print()

    pair_conditions = [c for c in by_condition if c != "solo"]
    if pair_conditions:
        print("=== pairs ===")
        summaries = [summarize_condition(c, by_condition[c]) for c in pair_conditions]
        print_pairs_table(summaries)
        print_same_role_roles(records)


if __name__ == "__main__":
    main()
