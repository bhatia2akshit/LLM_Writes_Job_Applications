from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.tools import tool
from pypdf import PdfReader
from prompt_service import PromptService

from env_loader import load_env_file

load_env_file(".env.local")

try:
    from litellm import completion
except Exception as e:  # pragma: no cover
    completion = None  # type: ignore[assignment]
    _LITELLM_IMPORT_ERROR = e


def _extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    return "\n\n".join(parts).strip()


def _get_model(model: str | None) -> str:
    if model:
        return model
    env_model = os.environ.get("LITELLM_MODEL")
    if env_model:
        return env_model
    if os.environ.get("MISTRAL_API_KEY"):
        return "mistral/mistral-small-latest"
    return "gpt-4o-mini"


def _litellm_chat(messages: list[dict[str, str]], model: str | None = None) -> str:
    if completion is None:  # pragma: no cover
        raise ImportError(
            "litellm is required but could not be imported. Install it with `pip install litellm`."
        ) from _LITELLM_IMPORT_ERROR

    response = completion(
        model=_get_model(model),
        messages=messages,
        temperature=0,
    )
    try:
        return response["choices"][0]["message"]["content"]
    except Exception:  # pragma: no cover
        return str(response)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_jsonish(text: str) -> dict[str, Any]:
    """
    Best-effort JSON object parsing for LLM outputs.
    Always returns a dict (either parsed JSON or a wrapper with the raw output).
    """
    candidate = text.strip()

    fence_match = _JSON_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
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
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            pass

    return {"llm_output": text}


@tool
def extract_pdf_pages(file_path: str) -> dict:

    """
    Extracts text from a PDF resume, then uses an LLM prompt to convert it into a dict.

    Parameters:
    - file_path (str): The path to the PDF file to be read.

    Returns:
    - dict: Parsed JSON output from the LLM (or a wrapper with the raw output).
    """
    prompt_name: str = "resume_to_dict"
    resume_text = _extract_pdf_text(file_path)
    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(resume_text=resume_text)
    llm_output = _litellm_chat(messages=messages)
    parsed = _parse_jsonish(llm_output)
    if "llm_output" in parsed:
        parsed["_resume_text"] = resume_text
    return parsed


@tool
def compare_cv_data(
    text_content,
    job_description: str,
    model: str = None,
    prompt_name: str = "compare_cv_to_job",
) -> dict:
    """
    Compares the extracted resume content with a job description using a prompt.

    Parameters:
    - text_content (str | dict): The extracted text content (or parsed resume dict).
    - job_description (str): The job description to compare against.

    Returns:
    - dict: A dictionary containing the comparison results.

    """
    resume_input = (
        json.dumps(text_content, ensure_ascii=False) if isinstance(text_content, dict) else str(text_content)
    )
    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(resume_input=resume_input, job_description=job_description)
    llm_output = _litellm_chat(messages=messages, model=model)
    parsed = _parse_jsonish(llm_output)
    if "llm_output" in parsed:
        parsed["_resume_input"] = resume_input
        parsed["_job_description"] = job_description
    return parsed

@tool
def write_cover_letter(
    text_content,
    job_description: str,
    model: str = None,
    prompt_name: str = "cover_letter",
) -> dict:
    """
    Generates a cover letter based on the resume content and job description using a prompt.

    Parameters:
    - text_content (str | dict): The extracted text content (or parsed resume dict).
    - job_description (str): The job description to tailor the cover letter to.

    Returns:
    - dict: Parsed JSON output from the LLM (or a wrapper with the raw output).
    """
    resume_input = (
        json.dumps(text_content, ensure_ascii=False) if isinstance(text_content, dict) else str(text_content)
    )
    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(resume_input=resume_input, job_description=job_description)
    llm_output = _litellm_chat(messages=messages, model=model)
    parsed = _parse_jsonish(llm_output)
    if "llm_output" in parsed:
        parsed["_resume_input"] = resume_input
        parsed["_job_description"] = job_description
    return parsed
    

if __name__ == "__main__":
    resume_data = extract_pdf_pages("CV_Aksh.pdf")
    print(json.dumps(resume_data, indent=2, ensure_ascii=False))
