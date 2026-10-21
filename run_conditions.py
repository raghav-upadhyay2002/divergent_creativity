from datetime import datetime, timezone

import conversation
import llm
import results

RESULTS_FILE = "pairs.jsonl"

CONDITIONS = [
    ("no_roles", None, None),
    ("different_roles", "product manager", "programmer"),
    ("same_role", "product manager", "product manager"),
]


def run_once(run_id, condition, role_a, role_b):
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


def run_condition(condition, role_a, role_b, n_runs=3):
    records = []
    for i in range(n_runs):
        run_id = f"{condition}-{i:03d}"
        try:
            record = run_once(run_id, condition, role_a, role_b)
        except llm.DailyLimitReached:
            print(f"Daily limit reached during condition '{condition}'. "
                  f"Saved {len(records)} run(s) before stopping.")
            break
        records.append(record)
    return records


def _step(record, step_name):
    return next((s for s in record["steps"] if s["step"] == step_name), None)


def summarize_condition(condition, records):
    exact_copy_count = 0
    exact_copy_total = 0

    start_scores = []
    final_scores = []
    final_shared_counts = []
    final_shared_words_union = set()

    for record in records:
        for s in record["steps"]:
            for label in ("agent_a", "agent_b"):
                agent_record = s[label]
                if agent_record and agent_record["comparison"] is not None:
                    exact_copy_total += 1
                    if agent_record["comparison"]["exact_copy"]:
                        exact_copy_count += 1

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
        "exact_copy_count": exact_copy_count,
        "exact_copy_total": exact_copy_total,
        "avg_start_score": avg(start_scores),
        "avg_final_score": avg(final_scores),
        "avg_final_shared_words": avg(final_shared_counts),
        "final_shared_words_seen": sorted(final_shared_words_union),
    }


def print_summary(summaries):
    def fmt_exact_copy(s):
        if s["exact_copy_total"] == 0:
            return "n/a"
        pct = 100 * s["exact_copy_count"] / s["exact_copy_total"]
        return f"{s['exact_copy_count']}/{s['exact_copy_total']} ({pct:.0f}%)"

    headers = ("condition", "n", "exact copies (agent-steps)", "avg start score",
               "avg final score", "avg shared @final")
    rows = []
    for s in summaries:
        rows.append((
            s["condition"], s["n_runs"], fmt_exact_copy(s),
            f"{s['avg_start_score']:.2f}" if s["avg_start_score"] is not None else "-",
            f"{s['avg_final_score']:.2f}" if s["avg_final_score"] is not None else "-",
            f"{s['avg_final_shared_words']:.1f}" if s["avg_final_shared_words"] is not None else "-",
        ))

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


def main():
    all_summaries = []
    for condition, role_a, role_b in CONDITIONS:
        records = run_condition(condition, role_a, role_b, n_runs=3)
        summary = summarize_condition(condition, records)
        all_summaries.append(summary)

    print_summary(all_summaries)
    return all_summaries


if __name__ == "__main__":
    main()
