import os
from typing import Dict
try:
    from langchain.agents import create_agent
except Exception:  # pragma: no cover - fallback for older/newer LangChain APIs
    from langchain.agents import initialize_agent

    def create_agent(*, model, tools, system_prompt):
        return initialize_agent(
            tools=tools,
            llm=model,
            agent="chat-zero-shot-react-description",
            verbose=False,
            agent_kwargs={"system_message": system_prompt},
        )
from helper import add_token_usage
from llm_service import LLMService
from tools import compare_cv_data, extract_cv_information, extract_jd_keywords, optimize_cv, write_cover_letter, write_new_cv

from pathlib import Path
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if _HF_TOKEN:
    login(token=_HF_TOKEN)
else:
    print("agent: HUGGINGFACEHUB_API_TOKEN not set; skipping login")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"Pydantic serializer warnings:")


available_tools = [extract_cv_information, extract_jd_keywords, compare_cv_data, optimize_cv]

def call_agent(cv_text: str, job_description: str):
    """
    Create and return a LangChain agent for CV extraction.
    
    The agent uses HuggingFace LLM and has access to the CV extraction, comparison, and optimization tools.
    """
    
    print("agent: init llm")
    # Initialize the main LLM for the agent
    main_llm = LLMService('huggingface').get_llm()

    print("agent: create agent")
    # Create the agent with the extraction tool
    agent = create_agent(
        model=main_llm,
        tools=available_tools,
        system_prompt="""You are a helpful CV/resume creating assistant.

          For the given text of CV and Job Description, you will create a new CV that is optimized for the job description.
          Follow the workflow below and do not skip any steps.

          Mandatory workflow (do not skip steps):
          1. Call the tool extract_cv_information tool on the CV text.
          2. Call the tool extract_jd_keywords tool on the job description text.
          3. Call the tool compare_cv_data tool using outputs from steps 1 and 2.
          4. Call the tool optimize_cv tool using outputs from step 1 and step 3.

          Requirements:
          - Always call ALL four tools in order before producing the final response.
          - Do not stop after a single tool call.
          - Use tool outputs as inputs to the next tool.
          - Do not manually parse, compare, or optimize outside tools.
          - Final answer must be the output of Optimize CV tool."""
      )
    print("agent: build prompt")
    prompt_text = (
        "Please follow the system defined rules for the following inputs.\n\n"
        f"CV:\n{cv_text}\n\n"
        f"Job Description:\n{job_description}"
    )
    add_token_usage("agent_prompt", prompt_text)
    print("agent: invoking")
    response = agent.invoke({
    "messages": [
        {
            "role": "user", 
            "content": prompt_text
        }
      ]
    })
    print(response)
    final_message = next(
    (m for m in reversed(response["messages"]) if getattr(m, "role", None) == "assistant" and getattr(m, "content", None)),
    None
)
    response_text = final_message.content if final_message else ""

    
    # print("agent: complete\n ", response)
    # response_text = response['messages'][1].content
    add_token_usage("agent_response", response_text)
    return response



def convert_latex_to_pdf(latex_content: str) -> bytes:
    # Convert LaTeX content to PDF
    pdf_bytes = b"%PDF-1.4..."
    return pdf_bytes
