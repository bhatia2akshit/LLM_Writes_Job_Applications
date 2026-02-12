from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any
from langchain_core.prompts import PromptTemplate
from typer import prompt

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = e


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str

    def render_messages(self, **variables: Any) -> list[dict[str, str]]:
        system = Template(self.system).safe_substitute(**variables).strip()
        user = Template(self.user).safe_substitute(**variables).strip()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages


class PromptService:
    def __init__(self, prompt_path: str | os.PathLike[str] | None = None):
        if prompt_path is None:
            prompt_path = os.environ.get("PROMPT_YAML_PATH")
        if prompt_path is None:
            prompt_path = Path(__file__).resolve().parent / "prompt.yaml"
        self._prompt_path = Path(prompt_path)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if yaml is None:  # pragma: no cover
            raise ImportError(
                "pyyaml is required but could not be imported. Install it with `pip install pyyaml`."
            ) from _YAML_IMPORT_ERROR
        if not self._prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt YAML not found: {self._prompt_path}. "
                "Create it or set PROMPT_YAML_PATH."
            )
        with self._prompt_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict) or "prompts" not in data:
            raise ValueError(
                f"Invalid prompt YAML format in {self._prompt_path}; expected top-level 'prompts' mapping."
            )
        return data

    def get(self, name: str) -> Prompt:
        prompts = self._data.get("prompts")
        if not isinstance(prompts, dict) or name not in prompts:
            available = sorted(prompts.keys()) if isinstance(prompts, dict) else []
            raise KeyError(f"Prompt '{name}' not found. Available: {available}")

        entry = prompts[name]
        if isinstance(entry, str):
            return Prompt(system="", user=entry)
        if not isinstance(entry, dict):
            raise ValueError(f"Prompt '{name}' must be a string or a mapping with keys 'system' and 'user'.")
        system = entry.get("system", "")
        user = entry.get("user", "")
        if not isinstance(system, str) or not isinstance(user, str):
            raise ValueError(f"Prompt '{name}' entries 'system' and 'user' must be strings.")
        return Prompt(system=system, user=user)
