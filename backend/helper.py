import json
import os
import re
from typing import Any

from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from fastapi import File
from pypdf import PdfReader
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoTokenizer

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

from huggingface_hub import login

_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if _HF_TOKEN:
    login(token=_HF_TOKEN)
else:
    print("helper: HUGGINGFACEHUB_API_TOKEN not set; skipping login")

def extract_pdf_text(fileObj: File) -> str:
    print("helper: extract_pdf_text start")
    reader = PdfReader(fileObj)
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    text = "<next-page>".join(parts).strip()
    print(f"helper: extract_pdf_text complete pages={len(reader.pages)} chars={len(text)}")
    return text


_TOKENIZER = None
_TOKEN_USAGE: dict[str, int] = {}


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        model_id = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZER


def _estimate_tokens(text: str) -> int:
    try:
        tokenizer = _get_tokenizer()
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def reset_token_usage() -> None:
    _TOKEN_USAGE.clear()


def add_token_usage(label: str, text: str) -> int:
    normalized = text if isinstance(text, str) else str(text)
    count = _estimate_tokens(normalized)
    _TOKEN_USAGE[label] = _TOKEN_USAGE.get(label, 0) + count
    return count


def print_token_usage() -> None:
    total = sum(_TOKEN_USAGE.values()) if _TOKEN_USAGE else 0
    payload = {
        "event": "llm_token_usage",
        "total_tokens": total,
        "breakdown": dict(_TOKEN_USAGE),
    }
    print(json.dumps(payload, ensure_ascii=True))


def parse_llm_json(response_text: str) -> dict:
    """
    Parse an LLM response string into a JSON object.

    The input should already contain a JSON object that matches the target schema.
    """
    print("tool: parse_llm_json start")
    parsed = parse_jsonish(response_text)
    if "llm_output" in parsed:
        print("tool: parse_llm_json failed to parse JSON")
        return {"error": "could_not_parse_json", "raw": response_text}
    print("tool: parse_llm_json success")
    return parsed


def parse_jsonish(text: str) -> dict[str, Any]:
    """
    Best-effort JSON object parsing for LLM outputs.
    Always returns a dict (either parsed JSON or a wrapper with the raw output).
    """
    print("helper: parse_jsonish start")
    candidate = text.strip()
    for marker in ("Final JSON Output", "Final JSON Object"):
        marker_idx = candidate.find(marker)
        if marker_idx != -1:
            candidate = candidate[marker_idx + len(marker) :].strip()
            break
    _JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    fence_match = _JSON_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            print("helper: parse_jsonish success")
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                print("helper: parse_jsonish success from snippet")
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            pass

    print("helper: parse_jsonish failed")
    return {"llm_output": text}
