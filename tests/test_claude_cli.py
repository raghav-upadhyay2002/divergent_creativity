import json
import subprocess

import pytest

from dat import conversation, llm, storage

TOKEN = "sk-ant-oat01-SENTINEL-TOKEN-do-not-leak"
WORDS = "apple, river, galaxy, shovel, memory, tunnel, violin, anchor, desert, pillow"


class FakeCLI:
    """Stands in for subprocess.run: scripted replies, records every call."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append({"command": list(command), **kwargs})
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "9.9.9 (Claude Code)\n", "")
        reply = self.replies.pop(0) if self.replies else ok_reply()
        return subprocess.CompletedProcess(command, *reply)

    @property
    def runs(self):
        return [c for c in self.calls if c["command"][1] != "--version"]


def ok_reply(text=WORDS, session_id="sess-1", model="claude-opus-5-5"):
    data = {
        "is_error": False, "result": text, "session_id": session_id,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 800,
                  "cache_read_input_tokens": 50, "output_tokens": 7},
        "modelUsage": {model: {"inputTokens": 2}},
    }
    return 0, json.dumps(data), ""


def error_reply(text="boom", stderr=""):
    return 1, json.dumps({"is_error": True, "result": text, "session_id": "sess-x"}), stderr


@pytest.fixture
def cli(tmp_path, monkeypatch):
    token_file = tmp_path / "token.txt"
    token_file.write_text(TOKEN + "\n")
    config = {
        "provider": {
            "type": "claude_cli", "model": "claude-opus-5-5", "effort": "low",
            "system_prompt": "You are a helpful assistant.",
            "token_file": str(token_file), "run_dir": str(tmp_path / "runs"),
        },
        "experiment": {}, "paths": {"results_dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "claude.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("DAT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-be-removed")
    monkeypatch.setattr(llm, "_provider_config", None)
    monkeypatch.setattr(llm, "_claude_token", None)
    monkeypatch.setattr(llm, "_claude_version", None)
    monkeypatch.setattr(llm, "_total_tokens", 0)
    sleeps = []
    monkeypatch.setattr(llm, "_sleep", lambda s: sleeps.append(s))
    fake = FakeCLI()
    monkeypatch.setattr(llm.subprocess, "run", fake)
    fake.sleeps = sleeps
    fake.token_file = token_file
    return fake


def msg(text="hi"):
    return [{"role": "user", "content": text}]


def test_command_flags_first_and_resumed_turn(cli):
    first = llm.send(msg("one"))
    llm.send(msg("two"), session_id=first["session_id"])

    c1, c2 = (c["command"] for c in cli.runs)
    for command in (c1, c2):
        assert command[:3] == ["claude", "-p", command[2]]
        assert command[command.index("--model") + 1] == "claude-opus-5-5"
        assert command[command.index("--effort") + 1] == "low"
        assert command[command.index("--system-prompt") + 1] == "You are a helpful assistant."
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--output-format") + 1] == "json"
        assert "--bare" not in command
        assert "--safe-mode" in command and "--strict-mcp-config" in command
    assert "--resume" not in c1
    assert c2[c2.index("--resume") + 1] == "sess-1"
    assert c2[2] == "two"  # only the latest message is sent on a resumed call


def test_token_only_in_env_and_api_key_removed(cli):
    llm.send(msg())
    call = cli.runs[0]
    assert call["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert not any(TOKEN in part for part in call["command"])
    assert call["cwd"] == llm._claude_run_dir()


def test_missing_or_empty_token_raises_and_runs_nothing(cli):
    cli.token_file.write_text("  \n")
    with pytest.raises(llm.ClaudeTokenError):
        llm.send(msg())
    cli.token_file.unlink()
    llm._claude_token = None
    with pytest.raises(llm.ClaudeTokenError):
        llm.send(msg())
    assert cli.runs == []


def test_parsed_fields_and_version_checked_once(cli):
    first = llm.send(msg())
    second = llm.send(msg(), session_id="sess-1")

    assert first["content"] == WORDS
    assert first["model"] == "claude-opus-5-5"
    assert first["effort"] == "low"
    assert first["temperature"] is None
    assert first["claude_code_version"] == "9.9.9 (Claude Code)"
    assert first["usage"]["prompt_tokens"] == 852
    assert first["usage"]["completion_tokens"] == 7
    assert first["usage"]["cache_read_input_tokens"] == 50
    assert llm.total_tokens_used() == 2 * 859
    assert second["claude_code_version"] == first["claude_code_version"]
    assert len([c for c in cli.calls if c["command"][1] == "--version"]) == 1


def test_is_error_is_retried_then_succeeds(cli):
    cli.replies = [error_reply("API Error: 500"), ok_reply()]
    result = llm.send(msg())
    assert result["content"] == WORDS
    assert result["failed_calls"] == ["API Error: 500"]
    assert result["failed_calls_on_resumed_session"] == 0
    assert cli.sleeps == [2]


def test_gives_up_after_three_network_style_failures(cli):
    cli.replies = [error_reply()] * 3
    with pytest.raises(llm.ClaudeCLIError):
        llm.send(msg())
    assert len(cli.runs) == 3


def test_rate_limit_backs_off_and_failed_resume_is_recorded(cli):
    cli.replies = [error_reply("API Error: 429 rate limit"), error_reply("overloaded"),
                   ok_reply()]
    result = llm.send(msg(), session_id="sess-1")
    assert cli.sleeps == [2, 4]
    assert result["failed_calls_on_resumed_session"] == 2  # duplicate user msgs in session


def test_gives_up_after_eight_rate_limits(cli):
    cli.replies = [error_reply("429 rate limit")] * 8
    with pytest.raises(llm.ClaudeCLIError):
        llm.send(msg())
    assert len(cli.runs) == 8


def test_daily_limit_raises_daily_limit_reached(cli):
    cli.replies = [error_reply("Claude usage limit reached")]
    with pytest.raises(llm.DailyLimitReached):
        llm.send(msg())
    assert cli.sleeps == []


def test_bad_json_and_timeout_are_failed_attempts(cli, monkeypatch):
    cli.replies = [(0, "not json", "")]
    result = llm.send(msg())  # second call falls back to ok_reply
    assert len(result["failed_calls"]) == 1

    calls = {"n": 0}

    def timing_out(command, **kwargs):
        calls["n"] += 1
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "9.9.9", "")
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(llm.subprocess, "run", timing_out)
    with pytest.raises(llm.ClaudeCLIError):
        llm.send(msg())


def test_agent_keeps_its_own_session(cli, monkeypatch):
    cli.replies = [ok_reply(session_id="A1"), ok_reply(session_id="B1"),
                   ok_reply(session_id="A1"), ok_reply(session_id="B1")]
    a, b = conversation.Agent("A"), conversation.Agent("B")
    a.send("a1"), b.send("b1"), a.send("a2"), b.send("b2")
    resumed = [c["command"][c["command"].index("--resume") + 1]
               for c in cli.runs if "--resume" in c["command"]]
    assert resumed == ["A1", "B1"]


def test_token_never_appears_in_logs_output_errors_or_saved_results(
        cli, tmp_path, monkeypatch, caplog, capsys):
    # The CLI misbehaves and echoes the token in stderr and in its error text.
    leaky = error_reply(f"auth failed for {TOKEN}", stderr=f"debug: {TOKEN}")
    cli.replies = [leaky, ok_reply(), leaky, leaky, leaky]
    monkeypatch.setattr(conversation, "score_details", lambda w: (50.0, list(w), []))
    monkeypatch.setattr(conversation, "score_details_all10", lambda w: (50.0, list(w), []))
    caplog.set_level("DEBUG")

    result = conversation.run_solo()
    record = storage.build_record("solo-000", "solo", result)
    runs_path = str(tmp_path / "results" / "claude" / "runs.jsonl")
    storage.save_run(record, runs_path)

    # a call that fails for good: the exception text must be scrubbed too
    with pytest.raises(llm.ClaudeCLIError) as excinfo:
        llm.send(msg())
    llm.print_token_usage()

    out = capsys.readouterr()
    saved = open(runs_path).read()
    assert "[REDACTED]" in saved  # the failure was recorded, scrubbed
    for text in (saved, json.dumps(record), caplog.text, out.out, out.err,
                 str(excinfo.value), repr(excinfo.value)):
        assert TOKEN not in text
    assert "sk-ant-api-should-be-removed" not in saved
    assert record["settings"]["temperature"] is None
    assert record["settings"]["effort"] == "low"
    attempt = record["steps"][0]["agent_a"]["attempts"][0]
    assert attempt["model"] == "claude-opus-5-5"
    assert attempt["claude_code_version"] == "9.9.9 (Claude Code)"
