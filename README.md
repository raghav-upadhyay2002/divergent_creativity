# Divergent Creativity

This measures how creatively LLMs generate word lists, using the Divergent
Association Task (DAT, Wang/Olson method). Each agent is asked for 10 nouns
"as different from each other as possible"; creativity is scored as the mean
pairwise cosine distance between the words' GloVe embeddings, so higher means
more semantically spread out. The experiment compares a single agent working
alone against pairs of agents who see and react to each other's lists.

## Conditions

| condition | agents | roles |
|---|---|---|
| `solo` | 1 | none |
| `no_roles` | 2 | neither agent is given a role |
| `different_roles` | 2 | different roles (e.g. product manager + programmer) |
| `same_role` | 2 | the same role for both (alternates PM+PM / programmer+programmer across runs) |

## How one two-agent conversation works

1. **start** -- each agent generates its own first list of 10 nouns, independently.
2. **round 1, 2, 3** -- each agent sees its partner's latest list, comments on
   it in a sentence or two, and gives a revised list (it may keep, swap, or
   combine words from either list).
3. **final** -- after 3 rounds, each agent gives its team's final combined list.

A `solo` run is just step 1 -- one agent, one list, no partner.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your Groq API key in a `.env` file in the project root:

```
GROQ_API_KEY=...
```

Download the GloVe vectors referenced by `paths.glove` in `configs/*.json`
from the OSF archive this experiment is built on top of: **osf.io/a9v2t**.

## How to run

```
python run_experiment.py --config configs/qwen.json
python run_experiment.py --config configs/qwen.json --conditions solo --n 20
python summarize.py results/qwen
python show_run.py results/qwen/runs.jsonl same_role-000
```

`run_experiment.py` runs every condition in the config by default; `--conditions`
restricts it to a subset (e.g. just `solo`), and `--n` overrides how many runs
per condition. Results go to `results/<config name>/runs.jsonl`. Any run that
fails is moved to `results/<config name>/failures.jsonl` (kept, never deleted)
and retried automatically the next time you run the same command.

`summarize.py` prints the score/copying tables for a results folder.
`show_run.py` prints one run's full step-by-step transcript -- every prompt
sent and every reply received.

## Tests

```
python -m pytest tests/
```

No test calls the real API -- `tests/test_llm.py` uses a fake client to
exercise the retry/backoff logic.

## File map

- `README.md` -- this file.
- `requirements.txt` -- exact pinned dependency versions.
- `configs/gptoss.json`, `configs/qwen.json` -- one provider/model setup each.
- `configs/claude.json` -- runs the Claude Code CLI (`provider.type: "claude_cli"`) from
  `.claude_runs/` with the token read from `.venv/claude_team_token.txt` (env only, never logged).
  Isolation flags (`--safe-mode --strict-mcp-config --disable-slash-commands`) keep a call
  to ~850 tokens. Known caveats: the model still sees the account email and a small environment
  block (cwd, model name, date), and a failed call on a resumed session leaves a duplicate user
  message in it (counted per reply as `failed_calls_on_resumed_session`).
- `dat/config.py` -- loads a config file (default `configs/gptoss.json`, or `$DAT_CONFIG_PATH`).
- `dat/llm.py` -- sends a chat request with rate-limit/network retry and backoff.
- `dat/prompts.py` -- every prompt template used in the experiment.
- `dat/parsing.py` -- extracts words (and comments) from a model's reply.
- `dat/scoring.py` -- the DAT score itself (Wang/Olson method).
- `dat/conversation.py` -- runs one agent (`run_solo`) or a two-agent conversation (`run_pair`).
- `dat/storage.py` -- builds, saves, and resumes run records.
- `run_experiment.py` -- the one runner: all conditions, or a chosen subset.
- `summarize.py` -- prints summary tables for a results folder.
- `show_run.py` -- prints one run's full transcript.
- `tests/test_scoring.py`, `tests/test_parsing.py`, `tests/test_llm.py` -- pytest tests for each building block.
- `results/<config name>/runs.jsonl` -- one line per completed run.
- `results/<config name>/failures.jsonl` -- runs that failed, kept for inspection.
- `results/archive/` -- old one-off runs kept for reference, outside the active pipeline.

## What one saved record contains

Each line of `runs.jsonl` is one run: `run_id`, `condition`, `timestamp`,
`roles` (`{"A": ..., "B": ...}`, `None` for solo/no_roles), `settings` (model,
temperature, etc.), overall `status`, and a `steps` list. Each step (`start`,
`round_1..3`, `final`) has a `status` and, per agent: every `attempts` entry
(prompt, raw reply, finish_reason, token usage), the final `parsed_words`,
the up-to-7 `valid_words` used for the main `dat_score`, the up-to-10
`valid_words_all10` used for the secondary `score_all10`, and -- in the
two-agent rounds -- a `comparison` against the partner's and its own previous
list. Each step also has a `team_score` and `shared_words` between the two
agents. (Some runs made before this structure existed are saved flatter --
`show_run.py` and `summarize.py` both handle that older shape too.)

## Adding a new model

Copy an existing config, e.g. `cp configs/qwen.json configs/mymodel.json`,
and edit its `provider` block (`model`, and `reasoning_effort`/
`reasoning_format` if the model supports them -- only send what it accepts).
Then run `python run_experiment.py --config configs/mymodel.json`; results
land in `results/mymodel/` automatically.
