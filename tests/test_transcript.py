import json
import subprocess
import sys

import transcript

PROMPT = "Imagine you are X, give 10 nouns."
ROUND_A = "Your partner's latest list is:\nb1, b2\n\nImprove."
ROUND_B = "Your partner's latest list is:\na1, a2\n\nImprove."
INSPIRED = ("1. cat: ACCEPT - fine\n2. dog: REJECT - too close to cat\n"
            "3. fish: ACCEPT - ok\nWORDS: cat, fish")


def agent(prompt, reply, dat, comment=None, comparison=None, attempts=None):
    return {"prompt": prompt, "raw_reply": reply, "comment": comment, "dat_score": dat,
            "comparison": comparison, "attempts": attempts}


def make_run(run_id="no_roles-000", condition="no_roles", status="ok"):
    cmp = {"kept_from_own": 6, "taken_from_partner": 3, "brand_new": 1}
    return {
        "run_id": run_id, "condition": condition, "status": status,
        "roles": {"A": None, "B": None},
        "steps": [
            {"step": "start", "team_score": 86.256,
             "agent_a": agent(PROMPT, "a, b", 87.123),
             "agent_b": agent(PROMPT, "c, d", 85.4)},
            {"step": "round_1", "team_score": 88.0,
             "agent_a": agent(ROUND_A, "COMMENT: nice\nWORDS: a", 88.0,
                              comment="nice", comparison=cmp),
             "agent_b": agent(ROUND_B, INSPIRED, 87.0,
                              attempts=[{"prompt": "old", "raw_reply": "bad"},
                                        {"prompt": ROUND_B, "raw_reply": INSPIRED}])},
            {"step": "final", "team_score": 90.0,
             "agent_a": agent(ROUND_A, "a", 90.0, comparison=cmp),
             "agent_b": agent(ROUND_B, "b", 89.0, comparison=cmp)},
        ],
    }


def test_select_runs_skips_solo_and_not_ok():
    """Only ok, non-solo runs matching the filters are kept."""
    solo = {"run_id": "solo-000", "condition": "solo", "status": "ok"}
    runs = [make_run(), make_run("no_roles-001", status="failed"),
            make_run("same_role-000", "same_role"), solo]
    assert [r["run_id"] for r in transcript.select_runs(runs)] == ["no_roles-000", "same_role-000"]
    assert len(transcript.select_runs(runs, condition="no_roles")) == 1
    assert transcript.select_runs(runs, run_id="same_role-000")[0]["condition"] == "same_role"


def test_render_header_start_and_summary():
    """Header, start block, 2-decimal rounding and the final summary line."""
    text = transcript.render_run(make_run())
    assert text.startswith("# no_roles-000  (roles: A=None, B=None)")
    assert "Prompt B: same as A" in text
    assert "DAT: A=87.12  B=85.40  Team=86.26" in text
    assert text.rstrip().endswith("Team DAT 86.26 -> 90.00 (+3.74)")


def test_round_comment_and_comparison_counts():
    """Non-inspired replies show the comment and kept/took/new counts."""
    text = transcript.render_run(make_run())
    assert "A comment: nice" in text
    assert "A agreement: kept own 6, took from partner 3, new 1" in text


def test_round_accept_reject_and_partner_list_only():
    """ACCEPT/REJECT replies are counted, rejects listed, B prompt shows only partner list."""
    text = transcript.render_run(make_run())
    assert "B agreement: accepted 2, rejected 1" in text
    assert "rejected dog: too close to cat" in text
    assert "Prompt B: same as A except partner list: a1, a2" in text


def test_retried_uses_last_attempt():
    """With retries, the last attempt is shown and flagged."""
    text = transcript.render_run(make_run())
    assert "B said (retried):" in text
    assert "bad" not in text


def test_cli_writes_out_file(tmp_path):
    """The CLI reads runs.jsonl and writes Markdown with --out."""
    (tmp_path / "runs.jsonl").write_text(json.dumps(make_run()) + "\n")
    out = tmp_path / "t.md"
    subprocess.run([sys.executable, "transcript.py", str(tmp_path), "--run", "no_roles-000",
                    "--out", str(out)], check=True, capture_output=True)
    assert "# no_roles-000" in out.read_text()
