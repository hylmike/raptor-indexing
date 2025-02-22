from abc import ABC, abstractmethod

import tiktoken

from raptor.llm_models import BaseEmbeddingModel, OpenAIEmbeddingModel
from raptor.tree_objects import Node, Tree
from raptor.utils.logger import logger
from raptor.utils.utils import (
    distance_from_embeddings,
    get_embeddings,
    get_node_list,
    get_text,
    indices_of_nearest_neighbors_from_distances,
    reverse_mapping,
)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> str:
        pass


class TreeRetrieverConfig:
    def __init__(  # noqa: PLR0913
        self,
        qa_model_name: str = None,
        select_mode: str = None,
        top_k: int = None,
        threshold: float = None,
        embedding_model: BaseEmbeddingModel = None,
        num_layers: int = None,
        start_layer: int = None,
    ) -> None:
        encoding = tiktoken.encoding_for_model(qa_model_name)
        tokenizer = tiktoken.get_encoding(encoding)
        self.tokenizer = tokenizer

        if select_mode is None:
            select_mode = "top_k"
        if not isinstance(select_mode, str) or select_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "select_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.select_mode = select_mode

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError(
                    "start_layer must be an integer and at least 0"
                )
        self.start_layer = start_layer

    def log_config(self):
        config_log = f"""
        TreeRetrieverConfig:
            Tokenizer: {self.tokenizer}
            Select Mode: {self.select_mode}
            Top K: {self.top_k}
            Threshold: {self.threshold}
            Embedding Model: {self.embedding_model}
            Num Layers: {self.num_layers}
            Start Layer: {self.start_layer}
        """

        return config_log


class TreeRetriever(BaseRetriever):
    """
    The TreeRetriever class is responsible for retriever all relavent documents from vector database based on query question.
    """

    def __init__(self, config: TreeRetrieverConfig, tree: Tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if (
            config.num_layers is not None
            and config.num_layers > tree.num_layers + 1
        ):
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if (
            config.start_layer is not None
            and config.start_layer > tree.num_layers
        ):
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        self.tree = tree
        self.num_layers = (
            config.num_layers
            if config.num_layers is not None
            else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer
            if config.start_layer is not None
            else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError(
                "num_layers must be less than or equal to start_layer + 1"
            )

        self.tokenizer = config.tokenizer
        self.select_mode = config.select_mode
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.embedding_model = config.embedding_model

        self.tree_node_index_to_layer = reverse_mapping(
            self.tree.layer_to_nodes
        )

        logger.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> list[float]:
        """
        Generates embeddings for the given text using the specified embedding model.
        """

        return self.embedding_model.create_embedding(text)

    def retrieve_information_collapse_tree(
        self, query: str, top_k: int, max_tokens: int
    ) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.
        """

        query_embedding = self.create_embedding(query)
        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)
        embeddings = get_embeddings(node_list)
        distances = distance_from_embeddings(query_embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for index in indices[:top_k]:
            node = node_list[index]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: list[Node], query: str, num_layers: int
    ) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []
        node_list = current_nodes

        for layer in range(num_layers):
            embeddings = get_embeddings(node_list)
            distances = distance_from_embeddings(query_embedding, embeddings)
            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.select_mode == "threshold":
                best_indices = [
                    index
                    for index in indices
                    if distances[index] > self.threshold
                ]
            else:
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[index] for index in best_indices]
            selected_nodes.append(nodes_to_add)

            if layer != num_layers - 1:
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve(  # noqa: PLR0913
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 5000,
        collapse_tree: bool = True,
        return_layer_info: bool = False,
    ) -> str:
        """
        Queries the tree and returns the most relevant information.
        """

        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.tree.num_layers
        ):
            raise ValueError(
                "start_layer must be an integer between 0 and tree.num_layers"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError(
                "num_layers must be less than or equal to start_layer + 1"
            )

        if collapse_tree:
            logger.info("Using collapsed_tree")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(
                layer_nodes, query, num_layers
            )

        if return_layer_info:
            layer_information = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[
                            node.index
                        ],
                    }
                )

            return context, layer_information

        return context
