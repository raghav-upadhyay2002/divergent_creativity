import argparse
import json
import sys


def parse_args():
    """Parse the results file and run_id to display."""
    parser = argparse.ArgumentParser(description="Print one run's full transcript.")
    parser.add_argument("results_file", help="e.g. results/qwen/runs.jsonl")
    parser.add_argument("run_id", help="e.g. same_role-000")
    return parser.parse_args()


def find_run(path, run_id):
    """Find and return the record for run_id in a jsonl file, or None."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("run_id") == run_id:
                return record
    return None


def print_attempts(label, attempts):
    """Print every attempt (prompt, raw reply, finish_reason, usage) for one agent."""
    for i, attempt in enumerate(attempts, start=1):
        print(f"  {label} attempt {i}:")
        print(f"    prompt: {attempt.get('prompt')}")
        print(f"    raw reply: {attempt.get('raw_reply')!r}")
        print(f"    finish_reason: {attempt.get('finish_reason')}, usage: {attempt.get('usage')}")


def print_last_reply_only(label, prompt, raw_reply):
    """Print just the saved prompt/raw_reply for a record with no attempts list."""
    print(f"  {label}: (older record: individual attempts not saved)")
    print(f"  {label} prompt: {prompt}")
    print(f"  {label} raw reply: {raw_reply!r}")


def print_agent(label, agent):
    """Print one agent's attempts (or a fallback note), words, and score for a step."""
    if agent is None:
        print(f"  {label}: (not attempted)")
        return

    attempts = agent.get("attempts")
    if attempts:
        print_attempts(label, attempts)
    else:
        print_last_reply_only(label, agent.get("prompt"), agent.get("raw_reply"))

    print(f"  {label} status: {agent.get('status')}")
    if agent.get("comment"):
        print(f"  {label} comment: {agent.get('comment')}")
    print(f"  {label} parsed words: {agent.get('parsed_words')}")
    print(f"  {label} valid words: {agent.get('valid_words')}")
    print(f"  {label} dat_score: {agent.get('dat_score')}")


def print_steps(record):
    """Print every step of a run, agent by agent (the current storage schema)."""
    print(f"roles: {record.get('roles')}")
    for step in record.get("steps", []):
        print(f"\n[{step.get('step')}] status={step.get('status')}")
        print_agent("A", step.get("agent_a"))
        print_agent("B", step.get("agent_b"))
        if step.get("team_score") is not None:
            print(f"  team_score: {step['team_score']:.2f}")
        if step.get("shared_words"):
            print(f"  shared_words: {step['shared_words']}")


def print_flat_solo(record):
    """Print the old flat solo schema that predates the "steps" list."""
    print("\n[solo]")
    attempts = record.get("attempts")
    if attempts:
        print_attempts("", attempts)
    else:
        print_last_reply_only("", record.get("prompt"), record.get("raw_reply"))

    print(f"  status: {record.get('status')}")
    print(f"  parsed words: {record.get('parsed_words')}")
    print(f"  valid words: {record.get('valid_words')}")
    print(f"  dat_score: {record.get('dat_score')}")


def main():
    """Find one run by id and print its full transcript, step by step."""
    args = parse_args()
    record = find_run(args.results_file, args.run_id)
    if record is None:
        print(f"run_id {args.run_id!r} not found in {args.results_file}")
        sys.exit(1)

    print(f"run_id: {record.get('run_id')}  condition: {record.get('condition')}  "
          f"status: {record.get('status')}")

    if "steps" in record:
        print_steps(record)
    else:
        print_flat_solo(record)


if __name__ == "__main__":
    main()
