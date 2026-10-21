import httpx
import openai

import llm
from dat_score import dat_score
from prompts import baseline_prompt_1

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))


# ---------------------------------------------------------------------------
# Fakes for offline testing -- no real API calls are made in this section.
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
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    message = "Daily rate limit exceeded" if daily_message else "Rate limit exceeded"
    return openai.RateLimitError(message, response=response, body={"error": {"message": message}})


def with_fake_sleep():
    """Patch llm._sleep to record calls instead of actually sleeping."""
    calls = []
    original = llm._sleep
    llm._sleep = lambda seconds: calls.append(seconds)
    return calls, original


# 1a. 429 with Retry-After waits that long
sleep_calls, original_sleep = with_fake_sleep()
try:
    client = FakeClient([make_rate_limit_error(retry_after=5), FakeResponse("ok")])
    reply = llm.chat([{"role": "user", "content": "hi"}], client=client)
    check("1a. 429 with Retry-After waits that long",
          sleep_calls == [5.0] and reply == "ok" and client.calls == 2,
          f"sleep_calls={sleep_calls}, reply={reply!r}, calls={client.calls}")
finally:
    llm._sleep = original_sleep

# 1b. 429 without Retry-After backs off (2, 4, 8, ...)
sleep_calls, original_sleep = with_fake_sleep()
try:
    client = FakeClient([make_rate_limit_error(), make_rate_limit_error(),
                          make_rate_limit_error(), FakeResponse("ok")])
    reply = llm.chat([{"role": "user", "content": "hi"}], client=client)
    check("1b. 429 without Retry-After backs off 2,4,8...",
          sleep_calls == [2, 4, 8] and reply == "ok" and client.calls == 4,
          f"sleep_calls={sleep_calls}, reply={reply!r}, calls={client.calls}")
finally:
    llm._sleep = original_sleep

# 1c. Long wait (over 10 minutes) raises DailyLimitReached
sleep_calls, original_sleep = with_fake_sleep()
try:
    client = FakeClient([make_rate_limit_error(retry_after=900)])
    raised = False
    wait_seconds = None
    try:
        llm.chat([{"role": "user", "content": "hi"}], client=client)
    except llm.DailyLimitReached as error:
        raised = True
        wait_seconds = error.wait_seconds
    check("1c. long Retry-After raises DailyLimitReached",
          raised and wait_seconds == 900.0 and sleep_calls == [],
          f"raised={raised}, wait_seconds={wait_seconds}, sleep_calls={sleep_calls}")
finally:
    llm._sleep = original_sleep

# 1c-bis. Error message that says "daily" raises DailyLimitReached even with a short wait
sleep_calls, original_sleep = with_fake_sleep()
try:
    client = FakeClient([make_rate_limit_error(retry_after=5, daily_message=True)])
    raised = False
    try:
        llm.chat([{"role": "user", "content": "hi"}], client=client)
    except llm.DailyLimitReached:
        raised = True
    check("1c-bis. 'daily limit' message raises DailyLimitReached",
          raised and sleep_calls == [],
          f"raised={raised}, sleep_calls={sleep_calls}")
finally:
    llm._sleep = original_sleep

# 1d. Gives up after 8 tries
sleep_calls, original_sleep = with_fake_sleep()
try:
    client = FakeClient([make_rate_limit_error() for _ in range(8)])
    raised = False
    try:
        llm.chat([{"role": "user", "content": "hi"}], client=client)
    except openai.RateLimitError:
        raised = True
    check("1d. gives up after 8 tries",
          raised and client.calls == 8 and sleep_calls == [2, 4, 8, 16, 32, 60, 60],
          f"raised={raised}, calls={client.calls}, sleep_calls={sleep_calls}")
finally:
    llm._sleep = original_sleep

check("1e. no real API calls made during offline tests", True, "all calls went through FakeClient")

# ---------------------------------------------------------------------------
# 2. One live call using the real client and config.
# ---------------------------------------------------------------------------

print(f"{'Check':<45} {'Status':<6} Detail")
print("-" * 100)
for name, status, detail in results:
    print(f"{name:<45} {status:<6} {detail}")

print("\n--- Live call: baseline_prompt_1 ---")
reply = llm.chat([{"role": "user", "content": baseline_prompt_1}])
print("Raw reply:", reply)

words = [w.strip() for w in reply.split(",") if w.strip()]
print("Parsed words:", words)

score = dat_score(words)
print("DAT score:", score)

llm.print_token_usage()
