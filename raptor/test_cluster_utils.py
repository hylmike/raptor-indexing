import numpy as np
import pytest
from raptor.cluster_utils import (
    reduce_cluster_embeddings,
    get_optimal_clusters,
    gmm_cluster,
    perform_clustering,
    RaptorClustering,
)
from raptor.tree_objects import Node

@pytest.fixture
def embeddings():
    return np.random.rand(10, 256)

@pytest.fixture
def nodes(embeddings):
    return [Node(text=f"Node {i}", index=i, children=set(), embedding=embeddings[i]) for i in range(10)]

def test_reduce_cluster_embeddings(embeddings):
    reduced_embeddings = reduce_cluster_embeddings(embeddings, dim=8)
    assert reduced_embeddings.shape[1] == 8

def test_get_optimal_clusters(embeddings):
    optimal_clusters = get_optimal_clusters(embeddings)
    assert optimal_clusters > 0

def test_gmm_cluster(embeddings):
    labels, n_clusters = gmm_cluster(embeddings, threshold=0.1)
    assert len(labels) == len(embeddings)
    assert n_clusters > 0

def test_perform_clustering(embeddings):
    clusters = perform_clustering(embeddings, target_dim=8, threshold=0.1)
    assert len(clusters) == len(embeddings)

def test_raptor_clustering(nodes):
    clustering_algo = RaptorClustering()
    clusters = clustering_algo.perform_clustering(nodes, llm_model_name="gpt-4o-mini", reduced_dimensions=8)
    assert isinstance(clusters, list)
    assert len(clusters) > 0

