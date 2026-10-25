import argparse
import json
import os
import re
import sys

ACCEPT_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*:\s*(ACCEPT|REJECT)\b\s*[-–—:]?\s*(.*)$",
                       re.IGNORECASE)
PARTNER_RE = re.compile(r"(latest list is:\n)(.*)")


def parse_args():
    """Parse the results folder and run filters."""
    parser = argparse.ArgumentParser(description="Print readable transcripts of saved runs.")
    parser.add_argument("results_dir", help="e.g. results/claude (contains runs.jsonl)")
    parser.add_argument("--condition", help="only runs of this condition, e.g. no_roles")
    parser.add_argument("--run", help="only this run_id, e.g. no_roles-003")
    parser.add_argument("--out", help="write Markdown to this file instead of printing")
    return parser.parse_args()


def load_runs(results_dir):
    """Read every record from runs.jsonl in results_dir."""
    path = os.path.join(results_dir, "runs.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_runs(records, condition=None, run_id=None):
    """Keep ok, non-solo runs matching the filters."""
    selected = []
    for r in records:
        if r.get("status") != "ok" or r.get("condition") == "solo" or "steps" not in r:
            continue
        if condition and r.get("condition") != condition:
            continue
        if run_id and r.get("run_id") != run_id:
            continue
        selected.append(r)
    return selected


def last_attempt(agent):
    """Return (prompt, raw_reply, retried) from the last attempt of an agent."""
    attempts = agent.get("attempts") or []
    if attempts:
        last = attempts[-1]
        return last.get("prompt"), last.get("raw_reply"), len(attempts) > 1
    return agent.get("prompt"), agent.get("raw_reply"), False


def fmt(score):
    """Format a score to 2 decimals, or '-' if missing."""
    return "-" if score is None else f"{score:.2f}"


def partner_list(prompt):
    """Extract the partner's word list line from a prompt, or None."""
    m = PARTNER_RE.search(prompt or "")
    return m.group(2) if m else None


def describe_prompt_b(prompt_a, prompt_b):
    """Show B's prompt as 'same as A', just the partner list, or in full."""
    if prompt_b == prompt_a:
        return "same as A"
    if PARTNER_RE.sub(r"\1", prompt_a or "") == PARTNER_RE.sub(r"\1", prompt_b or ""):
        return f"same as A except partner list: {partner_list(prompt_b)}"
    return prompt_b


def parse_accept_reject(raw_reply):
    """Return (accepted_count, [(word, reason)] rejected) or None if no ACCEPT/REJECT lines."""
    accepted, rejected, found = 0, [], False
    for line in (raw_reply or "").splitlines():
        m = ACCEPT_RE.match(line)
        if not m:
            continue
        found = True
        if m.group(2).upper() == "ACCEPT":
            accepted += 1
        else:
            rejected.append((m.group(1), m.group(3)))
    return (accepted, rejected) if found else None


def comparison_text(agent):
    """'kept own N, took from partner N, new N', or None if not saved."""
    c = agent.get("comparison")
    if not c:
        return None
    return (f"kept own {c.get('kept_from_own')}, took from partner "
            f"{c.get('taken_from_partner')}, new {c.get('brand_new')}")


def agreement_lines(label, agent, raw_reply, with_agreement=True):
    """Lines describing how an agent related to its partner's list."""
    lines = []
    ar = parse_accept_reject(raw_reply) if with_agreement else None
    if ar:
        accepted, rejected = ar
        lines.append(f"{label} agreement: accepted {accepted}, rejected {len(rejected)}")
        for word, reason in rejected:
            lines.append(f"  - rejected {word}: {reason}")
    elif with_agreement:
        if agent.get("comment"):
            lines.append(f"{label} comment: {agent['comment']}")
        text = comparison_text(agent)
        if text:
            lines.append(f"{label} agreement: {text}")
    return lines


def step_title(step):
    """Heading text for a step."""
    name = step.get("step", "")
    if name == "start":
        return "Start"
    if name == "final":
        return "Final"
    return name.replace("_", " ").title()


def render_step(step):
    """Render one step as a list of lines."""
    a, b = step.get("agent_a"), step.get("agent_b")
    out = [f"## {step_title(step)}"]
    if a is None or b is None:
        out.append("(agent not attempted)")
        return out
    is_start = step.get("step") == "start"
    pa, ra, retried_a = last_attempt(a)
    pb, rb, retried_b = last_attempt(b)
    out.append(f"Prompt A: {pa}")
    out.append(f"Prompt B: {describe_prompt_b(pa, pb)}")
    out.append(f"A said{' (retried)' if retried_a else ''}: {ra}")
    out.extend(agreement_lines("A", a, ra, with_agreement=not is_start))
    out.append(f"B said{' (retried)' if retried_b else ''}: {rb}")
    out.extend(agreement_lines("B", b, rb, with_agreement=not is_start))
    out.append(f"DAT: A={fmt(a.get('dat_score'))}  B={fmt(b.get('dat_score'))}  "
               f"Team={fmt(step.get('team_score'))}")
    return out


def render_run(record):
    """Render a whole run as Markdown text."""
    roles = record.get("roles") or {}
    lines = [f"# {record['run_id']}  (roles: A={roles.get('A')}, B={roles.get('B')})", ""]
    steps = record.get("steps", [])
    for step in steps:
        lines.extend(render_step(step))
        lines.append("")
    start = next((s for s in steps if s.get("step") == "start"), None)
    final = next((s for s in steps if s.get("step") == "final"), None)
    if start and final and start.get("team_score") is not None \
            and final.get("team_score") is not None:
        change = final["team_score"] - start["team_score"]
        lines.append(f"Team DAT {fmt(start['team_score'])} -> {fmt(final['team_score'])} "
                     f"({change:+.2f})")
    return "\n".join(lines).rstrip() + "\n"


def main():
    """Print or save transcripts for the selected runs."""
    args = parse_args()
    runs = select_runs(load_runs(args.results_dir), args.condition, args.run)
    if not runs:
        print("no matching runs found")
        sys.exit(1)
    text = "\n\n".join(render_run(r) for r in runs)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {len(runs)} run(s) to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
