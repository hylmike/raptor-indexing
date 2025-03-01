import os

import pytest
from unittest.mock import patch, MagicMock
from raptor.llm_models import GPT4oQAModel, OpenAIEmbeddingModel, GPTSummaryModel

@pytest.fixture
def mock_openai_client():
    os.environ["OPENAI_API_KEY"] = "test_key"
    with patch('raptor.llm_models.OpenAI') as MockOpenAI:
        yield MockOpenAI

def test_gpt4o_qa_model_answer_question(mock_openai_client):
    mock_client = mock_openai_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test answer"))]
    )

    model = GPT4oQAModel()
    answer = model.answer_question("Test context", "Test question")
    assert answer == "Test answer"

def test_openai_embedding_model_create_embedding(mock_openai_client):
    mock_client = mock_openai_client.return_value
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    model = OpenAIEmbeddingModel()
    embedding = model.create_embedding("Test text")
    assert embedding == [0.1, 0.2, 0.3]

def test_gpt_summary_model_summarize(mock_openai_client):
    mock_client = mock_openai_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test summary"))]
    )

    model = GPTSummaryModel()
    summary = model.summarize("Test context")
    assert summary == "Test summary"