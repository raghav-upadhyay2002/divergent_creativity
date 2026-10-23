import argparse
import os
import random
from pathlib import Path

from dat import conversation, llm, storage
from dat.config import load_config


def parse_args():
    """Parse --config, --conditions, and --n from the command line."""
    parser = argparse.ArgumentParser(description="Run the DAT experiment.")
    parser.add_argument("--config", default="configs/gptoss.json",
                         help="Path to a config file, e.g. configs/qwen.json")
    parser.add_argument("--conditions", nargs="+", default=None,
                         help="Only run these conditions (default: all conditions in the config)")
    parser.add_argument("--n", type=int, default=None,
                         help="Override runs_per_condition from the config")
    return parser.parse_args()


def results_paths(config, config_path):
    """Return (runs_path, failures_path) for this config's results folder."""
    name = Path(config_path).stem
    results_dir = os.path.join(config["paths"]["results_dir"], name)
    return os.path.join(results_dir, "runs.jsonl"), os.path.join(results_dir, "failures.jsonl")


def build_run_specs(conditions, roles, runs_per_condition):
    """Build one run spec per (condition, run index); same_role alternates roles."""
    specs = []
    for condition in conditions:
        for i in range(runs_per_condition):
            run_id = f"{condition}-{i:03d}"

            if condition == "different_roles":
                role_a, role_b = roles[0], roles[1]
            elif condition == "same_role":
                role = roles[i % len(roles)]
                role_a, role_b = role, role
            elif condition in ("no_roles", "solo"):
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


def shuffled(specs, seed):
    """Shuffle specs with a fixed seed, so conditions are mixed in time reproducibly."""
    specs = list(specs)
    random.Random(seed).shuffle(specs)
    return specs


def resume(specs, runs_path, failures_path):
    """Move old failures out of the way; return the specs that still need to run."""
    run_ids = {s["run_id"] for s in specs}
    ok_ids, failed, keep_lines = storage.partition_by_status(runs_path, run_ids)
    if failed:
        storage.move_failed_records(runs_path, failures_path, failed, keep_lines)
    pending = [s for s in specs if s["run_id"] not in ok_ids]
    return pending, len(failed)


def run_one(spec, runs_path, prompt_set="original"):
    """Run one spec (solo or pair), save it, and return the saved record."""
    if spec["condition"] == "solo":
        result = conversation.run_solo(prompt_set=prompt_set)
    else:
        result = conversation.run_pair(spec["role_a"], spec["role_b"], prompt_set=prompt_set)

    record = storage.build_record(spec["run_id"], spec["condition"], result)
    storage.save_run(record, runs_path)
    return record


def main():
    """Build the run list for a config, resume where needed, and execute it."""
    args = parse_args()
    os.environ["DAT_CONFIG_PATH"] = args.config
    config = load_config(args.config)
    llm.startup_checks()
    experiment = config["experiment"]

    prompt_set = experiment.get("prompt_set", "original")
    if prompt_set not in ("original", "inspired"):
        raise SystemExit(f"Unknown prompt_set: {prompt_set!r}")

    conditions = args.conditions or experiment["conditions"]
    runs_per_condition = args.n if args.n is not None else experiment["runs_per_condition"]

    specs = build_run_specs(conditions, experiment["roles"], runs_per_condition)
    specs = shuffled(specs, experiment["random_seed"])

    runs_path, failures_path = results_paths(config, args.config)
    pending, n_moved = resume(specs, runs_path, failures_path)

    print(f"Config: {args.config} (prompt_set: {prompt_set})")
    print(f"Results: {runs_path}")
    if n_moved:
        print(f"Moved {n_moved} old failed record(s) to {failures_path} (kept, not deleted)")
    print(f"Total runs: {len(specs)}, already done: {len(specs) - len(pending)}, "
          f"pending: {len(pending)}")

    completed = 0
    for spec in pending:
        try:
            record = run_one(spec, runs_path, prompt_set)
        except llm.DailyLimitReached:
            print(f"Daily limit reached. Completed {completed} run(s) before stopping.")
            break
        completed += 1
        print(f"  {spec['run_id']} ({spec['condition']}) -> {record['status']}")

    return completed


if __name__ == "__main__":
    main()
