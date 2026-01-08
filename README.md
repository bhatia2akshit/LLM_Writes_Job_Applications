# Cheap Flight Finding Agent

## Fixing `error: externally-managed-environment` (PEP 668)

If you’re on macOS with Homebrew Python, `pip install` into the system Python is blocked. Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Run:

```bash
python3 pdfreader.py
```

Notes:
- Prompts are loaded from `prompt.yaml` (override path with `PROMPT_YAML_PATH`).
- `pdfreader.py` loads `.env.local` automatically (example: `MISTRAL_API_KEY=...`) and you can set `LITELLM_MODEL` (example: `mistral/mistral-small-latest`).
