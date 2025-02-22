from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

import tiktoken

from .llm_models import (
    BaseEmbeddingModel,
    OpenAIEmbeddingModel,
    BaseSummaryModel,
    GPTSummaryModel,
)
from .tree_objects import Node, Tree
from .utils.utils import (
    distance_from_embeddings,
    get_embeddings,
    indices_of_nearest_neighbors_from_distances,
    split_text_with_token_limits,
)
from .utils.logger import logger


class TreeBuilderConfig:
    """
    The TreeBuilderConfig class is responsible for configuring all settings related to RAPTOR indexing tree
    """

    def __init__(
        self,
        max_tokens=None,
        num_layers=None,
        select_mode=None,
        top_k=None,
        threshold=None,
        summary_model=None,
        summary_length=None,
        embedding_model=None,
    ):
        tokenizer = tiktoken.encoding_for_model(summary_model)
        self.tokenizer = tokenizer

        if max_tokens is None:
            max_tokens = 1024
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if select_mode is None:
            select_mode = "top_k"
        if select_mode not in ["top_k", "threshold"]:
            raise ValueError(
                "selection_mode must be either 'top_k' or 'threshold'"
            )
        self.selection_mode = select_mode

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, int | float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if summary_model is None:
            summary_model = GPTSummaryModel()
        if not isinstance(summary_model, BaseSummaryModel):
            raise ValueError(
                "summary_model must be instance of BaseSummaryModel"
            )
        self.summary_model = summary_model

        if summary_length is None:
            summary_length = 200
        self.summary_length = summary_length

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

    def log_config(self):
        config_log = f"""
        TreeBuilderConfig
            Tokenizer: {self.tokenizer}
            Max Tokens: {self.max_tokens}
            Num Layers: {self.num_layers}
            select_mode: {self.selection_mode}
            top_k: {self.top_k}
            Threshold: {self.threshold}
            Summary Model: {self.summary_model}
            Summary Length: {self.summary_length}
            Embedding Model: {self.embedding_model}
        """
        return config_log


class TreeBuilder:
    """
    The TreeBuilder class is responsible for building a hierarchical text abstraction
    structure, known as a "tree," using summary model and embedding model.
    """

    def __init__(self, config: TreeBuilderConfig) -> None:
        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.select_mode = config.selection_mode
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.summary_length = config.summary_length
        self.summary_model = config.summary_model
        self.embedding_model = config.embedding_model

        logger.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )

    def create_node(
        self, index: int, text: str, children_indices: set[int] | None = None
    ) -> tuple[int, Node]:
        """
        Creates a new node with the given index, text, and (optionally) children indices.
        """

        if children_indices is None:
            children_indices = set()

        embedding = self.embedding_model.create_embedding(text)
        node = Node(
            text=text,
            index=index,
            children_indices=children_indices,
            embedding=embedding,
        )

        return (index, node)

    def create_embedding(self, text: str) -> list[float]:
        """
        Generates embeddings for the given text using the specified embedding model.
        """

        return self.embedding_model.create_embedding(text)

    def summarize(self, context: str, max_tokens=1024) -> str:
        """
        Generates a summary of the input context using the specified summarization model.
        """

        return self.summary_model.summarize(context, max_tokens)

    def get_relevant_nodes(
        self, current_node: Node, list_nodes: list[Node]
    ) -> list[Node]:
        """
        Retrieves the top-k most relevant nodes to the current node from the list of nodes
        based on cosine distance in the embedding space.
        """

        embeddings = get_embeddings(list_nodes)
        distances = distance_from_embeddings(current_node.embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.select_mode == "top_k":
            best_indices = indices[: self.top_k]
        else:
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]

        nodes_to_add = [list_nodes[index] for index in best_indices]
        return nodes_to_add

    def multithread_create_leaf_nodes(
        self, chunks: list[str]
    ) -> dict[int, Node]:
        """
        Creates leaf nodes using multithreading from the given list of text chunks.
        """

        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }
            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(
        self, text: str, use_multithreading: bool = True
    ) -> Tree:
        """
        Builds a golden tree from the input text, optionally using multithreading.
        """

        chunks = split_text_with_token_limits(
            text, self.summary_model, self.max_tokens
        )
        logger.info("Creating leaf nodes")

        if use_multithreading:
            leaf_nodes = self.multithread_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, text in enumerate(chunks):
                _, node = self.create_node(index, text)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}
        logger.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logger.info("Building All Nodes")
        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(
            all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes
        )

        return tree

    @abstractmethod
    def construct_tree(
        self,
        current_level_nodes: dict[int, Node],
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        use_multithreading: bool = True,
    ) -> dict[int, Node]:
        """
        Constructs the hierarchical tree structure layer by layer by iteratively summarizing groups
        of relevant nodes and updating the current_level_nodes and all_tree_nodes dictionaries at each step.
        """
        pass
