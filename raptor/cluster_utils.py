from abc import ABC, abstractmethod

import numpy as np
import umap
from sklearn.mixture import GaussianMixture
import tiktoken

from .utils.logger import logger
from .tree_objects import Node

RANDOM_SEED = 57
TARGET_REDUCED_DIMENSIONS = 8


# embed = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256)


def reduce_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int | None = None,
    metric: str = "cosine",
):
    """Perform global / local dimensionality reduction on the embeddings using UMAP."""

    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)

    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 100,
    random_state: int = RANDOM_SEED,
):
    """Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model."""

    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []

    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    return n_clusters[np.argmin(bics)]


def gmm_cluster(
    embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED
):
    """Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold."""

    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probabilities = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probabilities]

    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, target_dim: int, threshold: float
) -> list[np.ndarray]:
    """Perform clustering using a Gaussian Mixture Model, and finally performing local clustering within each global cluster."""

    if len(embeddings) <= target_dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction and clustering
    reduced_embeddings_global = reduce_cluster_embeddings(
        embeddings, target_dim
    )
    global_clusters, n_global_clusters = gmm_cluster(
        reduced_embeddings_global, threshold
    )

    logger.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings = embeddings[
            np.array(
                [i in global_cluster for global_cluster in global_clusters]
            )
        ]

        logger.info(
            f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings)}"
        )

        if len(global_cluster_embeddings) == 0:
            continue

        if len(global_cluster_embeddings) <= target_dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = reduce_cluster_embeddings(
                global_cluster_embeddings, target_dim
            )
            local_clusters, n_local_clusters = gmm_cluster(
                reduced_embeddings_local, threshold
            )

            logger.info(
                f"Local Clusters in Global Cluster {i}: {n_local_clusters}"
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings = global_cluster_embeddings[
                np.array(
                    [j in local_cluster for local_cluster in local_clusters]
                )
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings[:, None]).all(-1)
            )[1]
            for index in indices:
                all_local_clusters[index] = np.append(
                    all_local_clusters[index], j + total_clusters
                )

        total_clusters += n_local_clusters
        logger.info(f"Total Clusters: {total_clusters}")

    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(
        self,
        nodes: list[Node],
        llm_model_name: str,
        reduced_dimensions: int,
        **kwargs
    ) -> list[list[int]]:
        pass


class RaptorClustering(ClusteringAlgorithm):
    def perform_clustering(  # noqa: PLR0913
        self,
        nodes: list[Node],
        llm_model_name: str,
        reduced_dimensions: int,
        max_length_in_cluster: int = 50000,
        threshold: int = 0.1,
    ):
        tokenizer = tiktoken.encoding_for_model(llm_model_name)
        text_embeddings_np = np.array([node.embedding for node in nodes])

        clusters = perform_clustering(
            text_embeddings_np, reduced_dimensions, threshold=threshold
        )

        node_clusters = []

        for label in np.unique(np.concatenate(clusters)):
            indices = [
                i for i, cluster in enumerate(clusters) if label in cluster
            ]

            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(
                [len(tokenizer.encode(text)) for text in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                logger.info(
                    f"reclustering cluster with {len(cluster_nodes)} nodes"
                )

                node_clusters.extend(
                    RaptorClustering.perform_clustering(
                        cluster_nodes,
                        llm_model_name,
                        reduced_dimensions,
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
