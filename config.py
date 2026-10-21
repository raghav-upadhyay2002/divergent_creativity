import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"
_KEY_PREFIXES = ("gsk_", "sk-")


def load_config(path=_CONFIG_PATH):
    with open(path) as f:
        config = json.load(f)

    api_key_env = config.get("provider", {}).get("api_key_env", "")
    if api_key_env.startswith(_KEY_PREFIXES):
        raise ValueError(
            f"config.json provider.api_key_env looks like a real API key, not an "
            f"environment variable name. Put only the variable name there (e.g. "
            f"\"GROQ_API_KEY\"), and set the actual key via that environment variable."
        )

    return config


if __name__ == "__main__":
    print(json.dumps(load_config(), indent=2))
