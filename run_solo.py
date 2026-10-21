import argparse
import os
import statistics
from collections import Counter

import conversation
import llm
import parsing
import results
from dat_score import score_details
from prompts import baseline_prompt_1

CONDITION = "solo"
RESULTS_FILE = "solo.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the solo DAT baseline.")
    parser.add_argument("--config", default="config.json",
                         help="Path to the config file to use (default: config.json)")
    parser.add_argument("--out", default="solo.jsonl",
                         help="Results filename inside paths.results_dir (default: solo.jsonl)")
    parser.add_argument("--n", type=int, default=20,
                         help="Number of runs (default: 20)")
    return parser.parse_args()


def run_once(run_id, filename=RESULTS_FILE, client=None):
    agent = conversation.Agent("solo", client=client)
    result = conversation.attempt_step(agent, baseline_prompt_1, parsing.parse_list)

    if result["status"] != "ok":
        record = results.build_record(
            run_id=run_id, condition=CONDITION, prompt=baseline_prompt_1,
            raw_reply=result["raw_reply"], parsed_words=result["parsed_words"],
            valid_words=[], invalid_words=[], score=None, status=result["status"],
            attempts=result["attempts"], retries_used=result["retries_used"],
            reminders_used=result["reminders_used"])
        results.save_run(record, filename)
        return record

    score, valid_words, invalid_words = score_details(result["parsed_words"])
    status = "ok" if score is not None else "too_few_valid"

    record = results.build_record(
        run_id=run_id, condition=CONDITION, prompt=baseline_prompt_1,
        raw_reply=result["raw_reply"], parsed_words=result["parsed_words"],
        valid_words=valid_words, invalid_words=invalid_words, score=score, status=status,
        attempts=result["attempts"], retries_used=result["retries_used"],
        reminders_used=result["reminders_used"])
    results.save_run(record, filename)
    return record


def main(n_runs=20, filename=RESULTS_FILE):
    records = []
    for i in range(n_runs):
        run_id = f"solo-{i:03d}"
        try:
            record = run_once(run_id, filename=filename)
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
    args = parse_args()
    os.environ["DAT_CONFIG_PATH"] = args.config
    main(n_runs=args.n, filename=args.out)
