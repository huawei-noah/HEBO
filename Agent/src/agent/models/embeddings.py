from abc import ABC
from abc import abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    def __init__(self, model_id, logger, **kwargs) -> None:
        self.model_id = model_id
        self.logger = logger

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generates an embedding for the given text.

        Args:
        text (str): The text to be embedded.

        Returns:
        list[float]: The embedding vector.
        """
        pass

    @abstractmethod
    def batch_embed_texts(self, texts: list[str]) -> list[float]:
        """Embeds a batch of texts.

        Args:
        texts (list of str): The texts to be embedded.

        Returns:
        list[float]: A 2D array of embeddings.
        """
        pass

    def calculate_cosine_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Calculates similarity between two embeddings.

        Args:
        embedding1 (list[float]): First embedding vector.
        embedding2 (list[float]): Second embedding vector.

        Returns:
        float: The similarity score.
        """
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity

    def save_embeddings(self, embeddings: dict[list[float], str]) -> None:
        """Saves embeddings to a database.

        Args:
        embeddings (list[float]): The embeddings to save.
        """
        raise NotImplementedError

    def reduce_dimensionality(self, embeddings: list[list[float]], dimensions: int = 2) -> list[float]:
        """Reduces the dimensionality of embeddings using PCA.

        Args:
        embeddings (list of list[float]): List of high-dimensional embeddings.
        dimensions (int): The number of dimensions to reduce to.

        Returns:
        list[float]: Embeddings with reduced dimensionality.
        """
        from sklearn.decomposition import PCA

        # Stack embeddings into a 2D array for PCA
        stacked_embeddings = np.vstack(embeddings)
        pca = PCA(n_components=dimensions)
        reduced_embeddings = pca.fit_transform(stacked_embeddings)
        return reduced_embeddings

    def visualize_embeddings(self, embeddings: list[list[float]], labels: list[str] | None = None) -> None:
        """Visualizes embeddings in 2D space with a matplotlib plot figure.

        Args:
        embeddings (list of list[float]): List of high-dimensional embeddings.
        labels (list of str, optional): Labels for each embedding point.
        """
        import matplotlib.pyplot as plt

        reduced_embeddings = self.reduce_dimensionality(embeddings, 2)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)

        if labels is not None:
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("2D Visualization of Embeddings")
        plt.show()
