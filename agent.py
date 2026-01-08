from langchain.agents import initialize_agent
from pdfreader import compare_cv_data, extract_pdf_pages, write_cover_letter
# from litellm import completion


# def _litellm_chat(messages: list[dict[str, str]], model: str | None = None) -> str:
#     response = completion(
#         model="mistral/mistral-small-latest",#_get_model(model),
#         messages=messages,
#         temperature=0,
#     )
#     try:
#         return response["choices"][0]["message"]["content"]
#     except Exception:  # pragma: no cover
#         return str(response)


agent = initialize_agent(
    [extract_pdf_pages, compare_cv_data, write_cover_letter],
    "mistral/mistral-small-latest",
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True)

description = "AI Engineer with 5 years of experience in developing scalable web applications."
agent.invoke(f"CV_Aksh.pdf and {description}")
