import json
import os
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "gptoss.json"
_KEY_PREFIXES = ("gsk_", "sk-")


def load_config(path=None):
    """Load a config file (default: configs/gptoss.json, or $DAT_CONFIG_PATH)."""
    if path is None:
        path = os.environ.get("DAT_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    with open(path) as f:
        config = json.load(f)

    api_key_env = config.get("provider", {}).get("api_key_env", "")
    if api_key_env.startswith(_KEY_PREFIXES):
        raise ValueError(
            f"{path}: provider.api_key_env looks like a real API key, not an "
            f"environment variable name. Put only the variable name there (e.g. "
            f"\"GROQ_API_KEY\"), and set the actual key via that environment variable."
        )

    return config


if __name__ == "__main__":
    print(json.dumps(load_config(), indent=2))
