from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from raptor.cluster_utils import ClusteringAlgorithm, RaptorClustering
from raptor.tree_builder import TreeBuilder, TreeBuilderConfig
from raptor.tree_objects import Node, Tree
from raptor.utils.utils import get_node_list, get_text
from raptor.utils.logger import logger


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        target_reduced_dimension=10,
        clustering_algorithm=RaptorClustering(),
        clustering_params={},
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_reduced_dimension = target_reduced_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Target Reduced Dimension: {self.target_reduced_dimension}
        Clustering Algorithm: {self.clustering_algorithm}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config: ClusterTreeConfig) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.target_reduced_dimension = config.target_reduced_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logger.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def process_cluster(
        self, cluster, new_level_nodes, next_node_index, summary_length, lock
    ):
        node_texts = get_text(cluster)

        summarized_text = self.summarize(
            context=node_texts, max_tokens=summary_length
        )

        logger.info(
            f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
        )

        _, new_parent_node = self.create_node(
            next_node_index, summary_length, {node.index for node in cluster}
        )

        with lock:
            new_level_nodes[next_node_index] = new_parent_node

    def construct_tree(
        self,
        current_level_nodes,
        all_tree_nodes,
        layer_to_nodes,
        use_multithreading=False,
    ) -> dict[int, Node]:
        logger.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        for layer in range(self.num_layers):
            new_level_nodes = {}

            logger.info(f"Constructing Layer {layer}")
            node_list_current_layer = get_node_list(current_level_nodes)

            if (
                len(node_list_current_layer)
                <= self.target_reduced_dimension + 1
            ):
                self.num_layers = layer
                logger.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.embedding_model,
                reduced_dimensions=self.target_reduced_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summary_length = self.summary_length
            logger.info(f"Summary Length: {summary_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            self.process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summary_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)
            else:
                for cluster in clusters:
                    self.process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summary_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

        return current_level_nodes
