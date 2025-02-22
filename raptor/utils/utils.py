import re

import numpy as np
import tiktoken
from scipy import spatial

from raptor.tree_objects import Node


def reverse_mapping(layer_to_nodes: dict[int, list[Node]]) -> dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer

    return node_to_layer


def split_text_with_token_limits(
    text: str, llm_model_name: str, max_tokens: int, overlap: int = 0
) -> list[str]:
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    """

    encoding = tiktoken.encoding_for_model(llm_model_name)
    tokenizer = tiktoken.get_encoding(encoding)

    # Split the text into sentences using multiple delimiter patterns
    delimiters = ["[^\.!\?]+[\.|!|\?]\s", "[^\.!\?]+[\.|!|\?]$", "[^\.!\?]+$"]
    regex_pattern = "|".join(delimiters)
    sentences = [sentence.strip() for sentence in re.findall(regex_pattern, text)]
    # Calculate the number of tokens for each sentence
    num_tokens = [
        len(tokenizer.encode(" " + sentence)) for sentence in sentences
    ]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, num_tokens):
        if not sentence.strip():
            continue
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            # filter out empty strings
            filtered_sub_sentences = [
                sub.strip() for sub in sub_sentences if sub.strip() != ""
            ]
            sub_token_counts = [
                len(tokenizer.encode(" " + sub_sentence))
                for sub_sentence in filtered_sub_sentences
            ]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(
                filtered_sub_sentences, sub_token_counts
            ):
                if sub_length + sub_token_count > max_tokens:
                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyway in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(
                            sub_token_counts[
                                max(0, len(sub_chunk) - overlap) : len(
                                    sub_chunk
                                )
                            ]
                        )

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(
                num_tokens[
                    max(0, len(current_chunk) - overlap) : len(current_chunk)
                ]
            )
            current_chunk.append(sentence)
            current_length += token_count

        else:
            current_chunk.append(sentence)
            current_length += token_count
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def distance_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
    distance_metric: str = "cosine",
) -> list[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.
    """

    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def get_node_list(node_dict: dict[int, Node]) -> list[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.
    """

    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]

    return node_list


def get_embeddings(node_list: list[Node]) -> list[list[float]]:
    """
    Extracts the embeddings of nodes from a list of nodes.
    """

    return [node.embedding for node in node_list]


def get_children(node_list: list[Node]) -> list[set[int]]:
    """
    Extracts the children of nodes from a list of nodes.
    """

    return [node.children for node in node_list]


def get_text(node_list: list[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.
    """

    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(
    distances: list[float],
) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.
    """

    return np.argsort(distances)
