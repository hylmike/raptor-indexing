from abc import ABC, abstractmethod
import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from raptor.utils.logger import logger


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context: str, question: str):
        pass


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str):
        pass


class BaseSummaryModel(ABC):
    @abstractmethod
    def summarize(self, context: str, max_tokens: int = 1024):
        pass


class GPT4oQAModel(BaseQAModel):
    def __init__(self, model="gpt-4o-mini"):
        """
        Initializes the GPT-4o-mini model with the specified model version.
        """
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    @retry(
        wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6)
    )
    def answer_question(
        self, context: str, question: str, stop_sequence: str = None
    ):
        """
        Answer user question with given retrieved context
        """

        prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. Keep the answer brief and concise.
Question: {question} 
Context: {context} 
Answer:
"""
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are question answering portal",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01,
                stop=[stop_sequence],
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            logger.exception(f"Failed to get completion from {self.model}: {e}")
            raise


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-3-large", dimensions=256):
        self.client = OpenAI()
        self.model = model
        # Only supported dimensions in text-embedding-3 and later models
        self.dimensions = dimensions

    @retry(
        wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6)
    )
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        try:
            res = self.client.embeddings.create(
                input=[text], model=self.model, dimensions=self.dimensions
            )
            if not res:
                return None

            return res.data[0].embedding
        except Exception as e:
            logger.exception(f"Failed to generate embedding: {e}")
            raise


class GPTSummaryModel(BaseSummaryModel):
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    @retry(
        wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6)
    )
    def summarize(self, context, max_tokens=1024, stop_sequence=None):
        summary_prompt = f"""Give a detailed summary of the given documentation, summary will be embedded and used for retrieval.

Documentation:
{context}
"""
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.01,
            )

            return res.choices[0].message.content
        except Exception as e:
            logger.exception(f"Failed to create summary: {e}")
            raise
