import httpx
import openai

from dat import llm

# ---------------------------------------------------------------------------
# Fakes for offline testing -- no real API calls are ever made in this file.
# ---------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, total_tokens, prompt_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = total_tokens - prompt_tokens
        self.total_tokens = total_tokens


class FakeResponse:
    def __init__(self, content="ok", total_tokens=10, finish_reason="stop"):
        self.choices = [FakeChoice(content, finish_reason)]
        self.usage = FakeUsage(total_tokens)


class FakeCompletions:
    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        effect = self.side_effects.pop(0) if self.side_effects else FakeResponse()
        if isinstance(effect, BaseException):
            raise effect
        return effect


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, side_effects):
        self.completions_obj = FakeCompletions(side_effects)
        self.chat = FakeChat(self.completions_obj)

    @property
    def calls(self):
        return self.completions_obj.calls


def make_rate_limit_error(retry_after=None, daily_message=False):
    """Build a fake 429 RateLimitError, optionally with a Retry-After header."""
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    message = "Daily rate limit exceeded" if daily_message else "Rate limit exceeded"
    return openai.RateLimitError(message, response=response, body={"error": {"message": message}})


def fake_sleep(monkeypatch):
    """Patch llm._sleep to record calls instead of actually sleeping."""
    calls = []
    monkeypatch.setattr(llm, "_sleep", lambda seconds: calls.append(seconds))
    return calls


def test_rate_limit_with_retry_after_waits_that_long(monkeypatch):
    """A 429 with a Retry-After header waits exactly that long, then succeeds."""
    sleep_calls = fake_sleep(monkeypatch)
    client = FakeClient([make_rate_limit_error(retry_after=5), FakeResponse("ok")])

    result = llm.send([{"role": "user", "content": "hi"}], client=client)

    assert sleep_calls == [5.0]
    assert result["content"] == "ok"
    assert client.calls == 2


def test_rate_limit_without_retry_after_backs_off(monkeypatch):
    """A 429 with no Retry-After backs off 2, 4, 8 seconds."""
    sleep_calls = fake_sleep(monkeypatch)
    client = FakeClient([make_rate_limit_error(), make_rate_limit_error(),
                          make_rate_limit_error(), FakeResponse("ok")])

    result = llm.send([{"role": "user", "content": "hi"}], client=client)

    assert sleep_calls == [2, 4, 8]
    assert result["content"] == "ok"
    assert client.calls == 4


def test_long_retry_after_raises_daily_limit_reached(monkeypatch):
    """A Retry-After over 10 minutes raises DailyLimitReached instead of waiting."""
    sleep_calls = fake_sleep(monkeypatch)
    client = FakeClient([make_rate_limit_error(retry_after=900)])

    try:
        llm.send([{"role": "user", "content": "hi"}], client=client)
        assert False, "expected DailyLimitReached"
    except llm.DailyLimitReached as error:
        assert error.wait_seconds == 900.0

    assert sleep_calls == []


def test_daily_limit_message_raises_even_with_short_wait(monkeypatch):
    """An error message that says "daily" raises DailyLimitReached regardless of wait."""
    sleep_calls = fake_sleep(monkeypatch)
    client = FakeClient([make_rate_limit_error(retry_after=5, daily_message=True)])

    try:
        llm.send([{"role": "user", "content": "hi"}], client=client)
        assert False, "expected DailyLimitReached"
    except llm.DailyLimitReached:
        pass

    assert sleep_calls == []


def test_gives_up_after_8_tries(monkeypatch):
    """After 8 straight 429s with no Retry-After, send() gives up and raises."""
    sleep_calls = fake_sleep(monkeypatch)
    client = FakeClient([make_rate_limit_error() for _ in range(8)])

    try:
        llm.send([{"role": "user", "content": "hi"}], client=client)
        assert False, "expected RateLimitError"
    except openai.RateLimitError:
        pass

    assert client.calls == 8
    assert sleep_calls == [2, 4, 8, 16, 32, 60, 60]


def test_send_returns_finish_reason_and_usage():
    """send() returns the model's finish_reason and token usage, not just content."""
    client = FakeClient([FakeResponse("apple, dog", total_tokens=42, finish_reason="stop")])

    result = llm.send([{"role": "user", "content": "hi"}], client=client)

    assert result["content"] == "apple, dog"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 42


def test_think_tags_are_stripped():
    """A <think>...</think> block is stripped out of the returned content."""
    client = FakeClient([FakeResponse("<think>pondering</think>apple, dog")])

    result = llm.send([{"role": "user", "content": "hi"}], client=client)

    assert result["content"] == "apple, dog"
