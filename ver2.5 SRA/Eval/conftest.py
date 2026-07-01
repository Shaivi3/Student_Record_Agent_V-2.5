import os
import re

from dotenv import load_dotenv
load_dotenv()

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


class OllamaJudge(DeepEvalBaseLLM):
    JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1:8b")

    _JSON_SYSTEM = (
        "You are an evaluation model used by DeepEval.\n"
        "Return ONLY valid JSON.\n"
        "Do not explain.\n"
        "Do not use markdown.\n"
        "Do not wrap the JSON in code fences.\n"
        "Output raw JSON only."
    )

    def __init__(self):
        self.model = ChatOllama(
            model=self.JUDGE_MODEL,
            temperature=0,
            num_predict=1024,
        )

    def load_model(self):
        return self.model

    def _clean_json(self, text: str) -> str:
        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.MULTILINE,
        ).strip()

        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return match.group(1)

        return text

    def generate(self, prompt: str, schema=None) -> str:
        messages = [
            SystemMessage(content=self._JSON_SYSTEM),
            HumanMessage(content=prompt),
        ]

        response = self.model.invoke(messages)
        raw = response.content if response.content else ""

        return self._clean_json(raw)

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"ollama/{self.JUDGE_MODEL}"


judge = OllamaJudge()