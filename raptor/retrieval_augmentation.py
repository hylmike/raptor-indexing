import pickle

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .llm_models import (
    BaseEmbeddingModel,
    BaseQAModel,
    GPT4oQAModel,
    BaseSummaryModel,
)
from .retrievers import TreeRetrieverConfig, TreeRetriever
from .tree_objects import Tree
from raptor.utils.logger import logger


class RetrievalAugmentationConfig:
    """
    A Retrieval Augmentation config class to allow user customize setting of RAG
    """

    def __init__(  # noqa: PLR0913
        self,
        tree_builder_config=None,
        tree_retriever_config=None,
        qa_model=None,
        embedding_model=None,
        summary_model=None,
        tree_retriever_select_mode="top_k",
        tree_retriever_threshold=0.5,
        tree_retriever_top_k=5,
        tree_retriever_num_layers=None,
        tree_retriever_start_layer=None,
        tree_builder_max_tokens=1024,
        tree_builder_num_layers=5,
        tree_builder_threshold=0.5,
        tree_builder_top_k=5,
        tree_builder_select_mode="top_k",
        tree_builder_summary_length=200,
    ):
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )

        if summary_model is not None and not isinstance(
            summary_model, BaseSummaryModel
        ):
            raise ValueError(
                "summary_model must be an instance of BaseSummaryModel"
            )

        if tree_builder_config is None:
            tree_builder_config = ClusterTreeConfig(
                max_tokens=tree_builder_max_tokens,
                num_layer=tree_builder_num_layers,
                select_mode=tree_builder_select_mode,
                top_k=tree_builder_top_k,
                threshold=tree_builder_threshold,
                summary_model=summary_model,
                summary_length=tree_builder_summary_length,
                embedding_model=embedding_model,
            )
        elif not isinstance(tree_builder_config, ClusterTreeConfig):
            raise ValueError(
                "tree_builder_config must be a direct instance of ClusterTreeConfig"
            )

        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                qa_model_name=qa_model,
                select_mode=tree_retriever_select_mode,
                top_k=tree_retriever_top_k,
                threshold=tree_retriever_threshold,
                embedding_model=embedding_model,
                num_layers=tree_retriever_num_layers,
                start_layer=tree_retriever_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT4oQAModel()

    def log_config(self):
        config_summary = f"""
        {self.tree_builder_config.log_config()}

        {self.tree_retriever_config.log_config()}

        QA Model: {self.qa_model}
        """
        return config_summary


class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=None, tree=None):
        if config is None:
            config = RetrievalAugmentationConfig()
        elif not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # If tree is a string with a pickle file
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError(
                        "The loaded object is not an instance of Tree"
                    )
            except Exception as e:
                raise ValueError(
                    f"Fialed to load tree from pick file {tree}: {e}"
                )
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        self.tree_builder = ClusterTreeBuilder(config.tree_builder_config)
        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        if self.tree is not None:
            self.retriever = TreeRetriever(
                self.tree_retriever_config, self.tree
            )
        else:
            self.retriever = None

        logger.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_documents(self, docs: str):
        """
        Adds documents to the tree and creates a TreeRetriever instance.
        """
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                self.add_to_existing(docs)
                return

        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def add_to_existing(self, docs: str):
        """
        Adds documents to the existing tree.
        """
        pass

    def retrieve(
        self,
        question: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 5,
        max_tokens: int = 10000,
        collapse_tree: bool = True,
        return_layer_info: bool = True,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_info,
        )

    def answer_question(
        self,
        question: str,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 10000,
        collapse_tree: bool = True,
        return_layer_info: bool = False,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.
        """
        context, layer_information = self.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            True,
        )

        answer = self.qa_model.answer_question(context, question)

        if return_layer_info:
            return answer, layer_information

        return answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logger.info(f"Tree successfully saved to {path}")
