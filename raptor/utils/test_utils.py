import pytest
import numpy as np
from unittest.mock import patch

import tiktoken

from raptor.utils.utils import (
    reverse_mapping,
    split_text_with_token_limits,
    distance_from_embeddings,
    get_node_list,
    get_embeddings,
    get_children,
    get_text,
    indices_of_nearest_neighbors_from_distances,
)
from raptor.tree_objects import Node


@pytest.fixture
def sample_nodes():
    return {
        0: Node("Hello world", 0, {1, 2}, [0.1, 0.2]),
        1: Node("This is a test", 1, {3}, [0.3, 0.4]),
        2: Node("Another sentence", 2, set(), [0.5, 0.6]),
    }


def test_reverse_mapping(sample_nodes):
    layer_to_nodes = {
        0: [sample_nodes[0]],
        1: [sample_nodes[1], sample_nodes[2]],
    }
    expected = {0: 0, 1: 1, 2: 1}
    assert reverse_mapping(layer_to_nodes) == expected

def test_split_text_with_token_limits():
    text = "Hello world. This is a test! Another sentence?"
    llm_model_name = "gpt-4o-mini"
    max_tokens = 5
    overlap = 1
    expected = ['Hello world.', 'Hello world. This is a test!', 'This is a test! Another sentence?']
    encoding = "cl100k_base"
    tokenizer = tiktoken.get_encoding(encoding)
    with patch('tiktoken.encoding_for_model', return_value=encoding), patch.object(tokenizer, "encode", return_value=[1, 2, 3, 4]):
        assert split_text_with_token_limits(text, llm_model_name, max_tokens, overlap) == expected


def test_distance_from_embeddings():
    query_embedding = [0.1, 0.2]
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    expected_l2_results = [0.0, 0.28284271247461906, 0.5656854249492381]
    expected_cosine_results = [0.0, 0.01613008990009246, 0.026582831666424167]
    assert np.allclose(
        distance_from_embeddings(query_embedding, embeddings, "L2"), expected_l2_results
    )
    assert np.allclose(
        distance_from_embeddings(query_embedding, embeddings, "cosine"),
        expected_cosine_results,
    )


def test_get_node_list(sample_nodes):
    expected = [sample_nodes[0], sample_nodes[1], sample_nodes[2]]
    assert get_node_list(sample_nodes) == expected


def test_get_embeddings(sample_nodes):
    node_list = [sample_nodes[0], sample_nodes[1], sample_nodes[2]]
    expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert get_embeddings(node_list) == expected


def test_get_children(sample_nodes):
    node_list = [sample_nodes[0], sample_nodes[1], sample_nodes[2]]
    expected = [{1, 2}, {3}, set()]
    assert get_children(node_list) == expected


def test_get_text(sample_nodes):
    node_list = [sample_nodes[0], sample_nodes[1], sample_nodes[2]]
    expected = "Hello world\n\nThis is a test\n\nAnother sentence\n\n"
    assert get_text(node_list) == expected


def test_indices_of_nearest_neighbors_from_distances():
    distances = [0.3, 0.1, 0.2]
    expected = np.array([1, 2, 0])
    assert np.array_equal(
        indices_of_nearest_neighbors_from_distances(distances), expected
    )
