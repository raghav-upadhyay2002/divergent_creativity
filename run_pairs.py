import sys
from datetime import datetime, timezone

import conversation
import results

CONDITION = "pairs"
RESULTS_FILE = "pairs.jsonl"


def run_once(run_id, role_a, role_b):
    result = conversation.run_pair(role_a, role_b)

    record = {
        "run_id": run_id,
        "condition": CONDITION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "roles": result["roles"],
        "settings": results.current_settings(),
        "steps": result["steps"],
        "total_reminders_used": result["total_reminders_used"],
        "status": result["status"],
    }
    results.save_run(record, RESULTS_FILE)
    return record


def print_table(steps):
    def fmt_list(record):
        return ", ".join(record["parsed_words"]) if record else "(failed)"

    def fmt_score(record):
        if record and record["dat_score"] is not None:
            return f"{record['dat_score']:.2f}"
        return "-"

    rows = []
    for s in steps:
        team = f"{s['team_score']:.2f}" if s["team_score"] is not None else "-"
        shared = ", ".join(s["shared_words"]) if s["shared_words"] else "-"
        rows.append((
            s["step"],
            fmt_list(s["agent_a"]),
            fmt_score(s["agent_a"]),
            fmt_list(s["agent_b"]),
            fmt_score(s["agent_b"]),
            team,
            shared,
        ))

    headers = ("step", "A list", "A score", "B list", "B score", "team score", "shared words")
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]

    def print_row(row):
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))

    print_row(headers)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print_row(row)


def print_transcript(steps):
    print("\n--- Transcript ---")
    for s in steps:
        print(f"\n[{s['step']}]")
        for label, record in (("A", s["agent_a"]), ("B", s["agent_b"])):
            if record is None:
                print(f"  {label}: (not attempted)")
                continue
            print(f"  {label} prompt:")
            print(f"    {record['prompt']}")
            reminder_note = f" (after {record['reminders_used']} reminder(s))" \
                if record["reminders_used"] else ""
            print(f"  {label} reply{reminder_note}:")
            print(f"    {record['raw_reply']}")


def main(role_a="product manager", role_b="programmer", run_id="pairs-000"):
    record = run_once(run_id, role_a, role_b)

    print(f"Run status: {record['status']} (reminders used: {record['total_reminders_used']})\n")
    print_table(record["steps"])
    print_transcript(record["steps"])

    return record


if __name__ == "__main__":
    role_a = sys.argv[1] if len(sys.argv) > 1 else "product manager"
    role_b = sys.argv[2] if len(sys.argv) > 2 else "programmer"
    main(role_a, role_b)
