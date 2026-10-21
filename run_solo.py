import statistics
import sys
from collections import Counter

import llm
import parsing
import results
from dat_score import score_details
from prompts import baseline_prompt_1

CONDITION = "solo"
RESULTS_FILE = "solo.jsonl"


def run_once(run_id, filename=RESULTS_FILE):
    reply = llm.chat([{"role": "user", "content": baseline_prompt_1}])

    if not reply or not reply.strip():
        record = results.build_record(
            run_id=run_id, condition=CONDITION, prompt=baseline_prompt_1,
            raw_reply=reply, parsed_words=[], valid_words=[], invalid_words=[],
            score=None, status="empty")
        results.save_run(record, filename)
        return record

    parsed = parsing.parse_list(reply)

    if parsed.status != "ok":
        record = results.build_record(
            run_id=run_id, condition=CONDITION, prompt=baseline_prompt_1,
            raw_reply=reply, parsed_words=parsed.words, valid_words=[], invalid_words=[],
            score=None, status="parse_failed")
        results.save_run(record, filename)
        return record

    score, valid_words, invalid_words = score_details(parsed.words)
    status = "ok" if score is not None else "too_few_valid"

    record = results.build_record(
        run_id=run_id, condition=CONDITION, prompt=baseline_prompt_1,
        raw_reply=reply, parsed_words=parsed.words, valid_words=valid_words,
        invalid_words=invalid_words, score=score, status=status)
    results.save_run(record, filename)
    return record


def main(n_runs=20):
    records = []
    for i in range(n_runs):
        run_id = f"solo-{i:03d}"
        try:
            record = run_once(run_id)
        except llm.DailyLimitReached:
            print(f"Daily limit reached. Saved {len(records)} run(s) before stopping.")
            break
        records.append(record)

    print_summary(records)
    return records


def print_summary(records):
    status_counts = Counter(r["status"] for r in records)

    print("\n--- Summary ---")
    print(f"Total runs: {len(records)}")
    for status in sorted(status_counts):
        print(f"  {status}: {status_counts[status]}")

    scores = [r["dat_score"] for r in records if r["status"] == "ok"]
    if scores:
        print(f"\nAverage score: {statistics.mean(scores):.2f}")
        sd = statistics.stdev(scores) if len(scores) > 1 else float("nan")
        print(f"SD: {sd:.2f}" if len(scores) > 1 else "SD: n/a (only one score)")
        print(f"Min: {min(scores):.2f}")
        print(f"Max: {max(scores):.2f}")
    else:
        print("\nNo successful scores to summarize.")

    all_words = []
    for r in records:
        all_words.extend(r["valid_words"])

    print(f"\nUnique words across all lists: {len(set(all_words))}")

    counts = Counter(all_words)
    print("\nTop 10 most common words:")
    for word, count in counts.most_common(10):
        print(f"  {word}: {count}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
