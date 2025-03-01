import pytest
from unittest.mock import patch

from raptor.llm_models import BaseEmbeddingModel, BaseSummaryModel
from raptor.tree_builder import TreeBuilder, TreeBuilderConfig
from raptor.tree_objects import Node, Tree

@pytest.fixture
def mock_embedding_model():
    with patch('raptor.tree_builder.OpenAIEmbeddingModel', spec=BaseEmbeddingModel) as MockEmbeddingModel:
        yield MockEmbeddingModel

@pytest.fixture
def mock_summary_model():
    with patch('raptor.tree_builder.GPTSummaryModel', spec=BaseSummaryModel) as MockSummaryModel:
        yield MockSummaryModel

@pytest.fixture
def tree_builder_config(mock_embedding_model, mock_summary_model):
    mock_summary_model.return_value.model = "gpt-4o-mini"
    return TreeBuilderConfig(
        max_tokens=1024,
        num_layers=5,
        select_mode="top_k",
        top_k=5,
        threshold=0.5,
        summary_model=mock_summary_model.return_value,
        summary_length=200,
        embedding_model=mock_embedding_model.return_value
    )

@pytest.fixture
def tree_builder(tree_builder_config):
    return TreeBuilder(tree_builder_config)

def test_create_node(tree_builder, mock_embedding_model):
    mock_embedding_model.return_value.create_embedding.return_value = [0.1, 0.2, 0.3]
    index, node = tree_builder.create_node(0, "Test text")
    assert index == 0
    assert node.text == "Test text"
    assert node.embedding == [0.1, 0.2, 0.3]

def test_create_embedding(tree_builder, mock_embedding_model):
    mock_embedding_model.return_value.create_embedding.return_value = [0.1, 0.2, 0.3]
    embedding = tree_builder.create_embedding("Test text")
    assert embedding == [0.1, 0.2, 0.3]

def test_summarize(tree_builder, mock_summary_model):
    mock_summary_model.return_value.summarize.return_value = "Test summary"
    summary = tree_builder.summarize("Test context")
    assert summary == "Test summary"

def test_get_relevant_nodes(tree_builder, mock_embedding_model):
    mock_embedding_model.return_value.create_embedding.return_value = [0.1, 0.2, 0.3]
    current_node = Node(text="Current node", index=0, children=set(), embedding=[0.1, 0.2, 0.3])
    list_nodes = [Node(text=f"Node {i}", index=i, children=set(), embedding=[0.1, 0.2, 0.3]) for i in range(5)]
    relevant_nodes = tree_builder.get_relevant_nodes(current_node, list_nodes)
    assert len(relevant_nodes) == 5

def test_multithread_create_leaf_nodes(tree_builder, mock_embedding_model):
    mock_embedding_model.return_value.create_embedding.return_value = [0.1, 0.2, 0.3]
    chunks = ["Text chunk 1", "Text chunk 2", "Text chunk 3"]
    leaf_nodes = tree_builder.multithread_create_leaf_nodes(chunks)
    assert len(leaf_nodes) == 3
    for index, node in leaf_nodes.items():
        assert node.text == chunks[index]

def test_build_from_text(tree_builder, mock_embedding_model):
    mock_embedding_model.return_value.create_embedding.return_value = [0.1, 0.2, 0.3]
    text = "Test text for building tree"
    tree = tree_builder.build_from_text(text)
    assert isinstance(tree, Tree)
    assert len(tree.leaf_nodes) > 0