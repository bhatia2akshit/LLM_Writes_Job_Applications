from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

class LLMService:
    def __init__(self, model_type: str="huggingface", model_id: str="Qwen/Qwen3-Coder-30B-A3B-Instruct"):#"openai/gpt-oss-20b"): #"meta-llama/Llama-3.1-8B-Instruct"):# 
        self.model_id = model_id
        if model_type == "mistral":
            self.llm = self.mistral_llm()
        elif model_type == "litellm":
            self.llm = self.lite_llm()
        else:
            self.llm = self.huggingface_llm()
    
    def get_llm(self):
        return self.llm

    def lite_llm(self):
        """Initialize Lite LLM"""
        print(f"helper: init_lite_llm model_id={self.model_id}")
        llm = ChatOpenAI(
            model=self.model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.litellm.ai"
        )
        return llm


    def huggingface_llm(self):
        """Initialize HuggingFace LLM"""
        print(f"helper: init_huggingface_llm model_id={self.model_id}")
        llm = HuggingFaceEndpoint(
            repo_id=self.model_id,
            task="text-generation",
            max_new_tokens=10012,
            do_sample=False,
            repetition_penalty=1.03,
            provider="auto",
        )

        # Wrap the LLM with ChatHuggingFace to enable tool calling
        chat_model = ChatHuggingFace(llm=llm)

        print("✓ HuggingFace ChatModel initialized")
        return chat_model

    def mistral_llm(self):
        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        return llm
