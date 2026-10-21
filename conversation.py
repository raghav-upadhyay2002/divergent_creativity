import llm
import parsing
import prompts
from dat_score import score_details, score_details_all10


class Agent:
    def __init__(self, name, role=None, client=None):
        self.name = name
        self.role = role
        self.client = client
        self.history = []

    def send(self, content):
        """Send content, return llm.send()'s rich dict (content/finish_reason/usage)."""
        self.history.append({"role": "user", "content": content})
        result = llm.send(self.history, client=self.client)
        self.history.append({"role": "assistant", "content": result["content"]})
        return result


def _words_to_text(words):
    return ", ".join(words)


def _norm(word):
    return word.strip().lower()


def _compare_lists(new_words, own_previous, partner_previous):
    """Compare a new list against the agent's own previous list and the
    partner's previous list (the one shown to the agent in its prompt)."""
    new_norm = [_norm(w) for w in new_words]
    own_prev_norm = [_norm(w) for w in own_previous]
    partner_prev_norm = [_norm(w) for w in partner_previous]

    own_prev_set = set(own_prev_norm)
    partner_prev_set = set(partner_prev_norm)

    kept_from_own = 0
    taken_from_partner = 0
    brand_new = 0
    for w in new_norm:
        if w in own_prev_set:
            kept_from_own += 1
        elif w in partner_prev_set:
            taken_from_partner += 1
        else:
            brand_new += 1

    return {
        "exact_copy": new_norm == partner_prev_norm,
        "same_words": set(new_norm) == partner_prev_set,
        "kept_from_own": kept_from_own,
        "taken_from_partner": taken_from_partner,
        "brand_new": brand_new,
    }


def attempt_step(agent, prompt, parse_fn):
    """Send prompt, parse reply, and recover from a bad first attempt.

    - Empty reply -> resend the SAME prompt once (a retry, not a reminder).
    - Non-empty but unreadable -> send the format reminder once.
    - Whichever path applies, if it still fails the specific parsing status
      ("empty", "too_few_words", "unreadable") is kept, not a generic failure code.

    Every attempt (prompt, raw reply, finish_reason, usage) is recorded and kept --
    the first reply is never overwritten/discarded.
    """
    attempts = []

    def _send(prompt_text):
        result = agent.send(prompt_text)
        attempts.append({
            "prompt": prompt_text,
            "raw_reply": result["content"],
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
        })
        return result["content"]

    reply = _send(prompt)
    parsed = parse_fn(reply)

    retries_used = 0
    reminders_used = 0

    if parsed.status == "empty":
        retries_used = 1
        reply = _send(prompt)  # resend the SAME prompt, not a reminder
        parsed = parse_fn(reply)
    elif parsed.status != "ok":
        reminders_used = 1
        reply = _send(prompts.reminder_prompt)
        parsed = parse_fn(reply)

    return {
        "prompt": prompt,
        "attempts": attempts,
        "raw_reply": reply,
        "comment": parsed.comment,
        "parsed_words": parsed.words,
        "status": parsed.status,  # "ok", "empty", "too_few_words", or "unreadable"
        "retries_used": retries_used,
        "reminders_used": reminders_used,
    }


def _agent_record(result, own_previous=None, partner_previous=None, own_previous_valid=None):
    if result is None:
        return None

    score, valid_words, invalid_words = score_details(result["parsed_words"])
    score_all10, _used_all10, _dropped_all10 = score_details_all10(result["parsed_words"])

    comparison = None
    if result["status"] == "ok" and own_previous is not None and partner_previous is not None:
        comparison = _compare_lists(result["parsed_words"], own_previous, partner_previous)

    words_changed_since_previous = None
    if result["status"] == "ok" and own_previous_valid is not None:
        words_changed_since_previous = len(set(valid_words) - set(own_previous_valid))

    return {
        "prompt": result["prompt"],
        "attempts": result["attempts"],
        "raw_reply": result["raw_reply"],
        "comment": result["comment"],
        "parsed_words": result["parsed_words"],
        "valid_words": valid_words,
        "invalid_words": invalid_words,
        "dat_score": score,
        "score_all10": score_all10,
        "words_changed_since_previous": words_changed_since_previous,
        "retries_used": result["retries_used"],
        "reminders_used": result["reminders_used"],
        "status": result["status"],
        "comparison": comparison,
    }


def _run_step(step_name, step_type, agent_a, agent_b, prompt_a, prompt_b, parse_fn,
              prev_words=None, prev_valid_words=None):
    own_prev_a = prev_words["A"] if prev_words else None
    partner_prev_a = prev_words["B"] if prev_words else None
    own_prev_b = prev_words["B"] if prev_words else None
    partner_prev_b = prev_words["A"] if prev_words else None

    own_prev_valid_a = prev_valid_words["A"] if prev_valid_words else None
    own_prev_valid_b = prev_valid_words["B"] if prev_valid_words else None

    result_a = attempt_step(agent_a, prompt_a, parse_fn)
    record_a = _agent_record(result_a, own_prev_a, partner_prev_a, own_prev_valid_a)

    if record_a["status"] != "ok":
        return _build_step(step_name, step_type, record_a, None)

    result_b = attempt_step(agent_b, prompt_b, parse_fn)
    record_b = _agent_record(result_b, own_prev_b, partner_prev_b, own_prev_valid_b)

    return _build_step(step_name, step_type, record_a, record_b)


def _build_step(step_name, step_type, record_a, record_b):
    if record_a["status"] != "ok":
        status = record_a["status"]
    elif record_b is None:
        status = "ok"
    elif record_b["status"] != "ok":
        status = record_b["status"]
    else:
        status = "ok"

    team_score = None
    if record_a and record_b and record_a["dat_score"] is not None \
            and record_b["dat_score"] is not None:
        team_score = (record_a["dat_score"] + record_b["dat_score"]) / 2

    shared_words = []
    if record_a and record_b:
        shared_words = sorted(set(record_a["valid_words"]) & set(record_b["valid_words"]))

    return {
        "step": step_name,
        "step_type": step_type,  # "start", "round", or "final" -- lets reporting
                                  # count copying separately for rounds vs. the final step
        "status": status,
        "agent_a": record_a,
        "agent_b": record_b,
        "team_score": team_score,
        "shared_words": shared_words,
    }


def run_pair(role_a, role_b, client=None):
    agent_a = Agent("A", role_a, client=client)
    agent_b = Agent("B", role_b, client=client)

    steps = []

    start_prompt_a = prompts.role_start_prompt(role_a) if role_a else prompts.baseline_prompt_1
    start_prompt_b = prompts.role_start_prompt(role_b) if role_b else prompts.baseline_prompt_1

    step = _run_step("start", "start", agent_a, agent_b, start_prompt_a, start_prompt_b,
                      parsing.parse_list)
    steps.append(step)

    if step["status"] == "ok":
        prev_a = step["agent_a"]["parsed_words"]
        prev_b = step["agent_b"]["parsed_words"]
        prev_valid_a = step["agent_a"]["valid_words"]
        prev_valid_b = step["agent_b"]["valid_words"]

        for round_n in (1, 2, 3):
            prompt_a = prompts.round_prompt_template.format(partner_list=_words_to_text(prev_b))
            prompt_b = prompts.round_prompt_template.format(partner_list=_words_to_text(prev_a))

            step = _run_step(f"round_{round_n}", "round", agent_a, agent_b, prompt_a, prompt_b,
                              parsing.parse_round,
                              prev_words={"A": prev_a, "B": prev_b},
                              prev_valid_words={"A": prev_valid_a, "B": prev_valid_b})
            steps.append(step)

            if step["status"] != "ok":
                break

            prev_a = step["agent_a"]["parsed_words"]
            prev_b = step["agent_b"]["parsed_words"]
            prev_valid_a = step["agent_a"]["valid_words"]
            prev_valid_b = step["agent_b"]["valid_words"]

        else:
            prompt_a = prompts.final_prompt_template.format(partner_list=_words_to_text(prev_b))
            prompt_b = prompts.final_prompt_template.format(partner_list=_words_to_text(prev_a))

            step = _run_step("final", "final", agent_a, agent_b, prompt_a, prompt_b,
                              parsing.parse_list,
                              prev_words={"A": prev_a, "B": prev_b},
                              prev_valid_words={"A": prev_valid_a, "B": prev_valid_b})
            steps.append(step)

    total_reminders = sum(
        (s["agent_a"]["reminders_used"] if s["agent_a"] else 0)
        + (s["agent_b"]["reminders_used"] if s["agent_b"] else 0)
        for s in steps
    )
    total_retries = sum(
        (s["agent_a"]["retries_used"] if s["agent_a"] else 0)
        + (s["agent_b"]["retries_used"] if s["agent_b"] else 0)
        for s in steps
    )
    overall_status = next((s["status"] for s in steps if s["status"] != "ok"), "ok")

    return {
        "roles": {"A": role_a, "B": role_b},
        "steps": steps,
        "total_reminders_used": total_reminders,
        "total_retries_used": total_retries,
        "status": overall_status,
    }
