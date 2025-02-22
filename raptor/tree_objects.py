class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(
        self,
        text: str,
        index: int,
        children: set[int],
        embedding: list[float],
    ) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embedding = embedding


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self,
        all_nodes: list[Node],
        root_nodes: list[Node],
        leaf_nodes: list[Node],
        num_layers: int,
        layer_to_nodes: dict[int, list[Node]],
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layers_to_nodes = layer_to_nodes
