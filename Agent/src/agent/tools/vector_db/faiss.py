# extension of https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L934
# to allow using FAISS with fschat model with our renaming

import pickle
import uuid
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
from langchain_community.docstore.base import AddableMixin
from langchain_community.docstore.base import Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import _len_check_if_sized
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FAISS_DB(FAISS):
    def __init__(
        self,
        embedding_function,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        if self.distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE and self._normalize_L2:
            warnings.warn(f"Normalizing L2 is not applicable for metric type: {self.distance_strategy}")

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding_function

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embedding_function(text) for text in texts]

    def _embed_query(self, text: str) -> List[float]:
        return self.embedding_function(text)

    @classmethod
    def __from(
        cls,
        texts: Iterable[str],
        embeddings: List[List[float]],
        embedding_function: Any,
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(embeddings[0]))
        vecstore = cls(
            embedding_function,
            index,
            InMemoryDocstore(),
            {},
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )
        vecstore.__add(texts, embeddings, metadatas=metadatas, ids=ids)
        return vecstore

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_function: Any,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        embeddings = [embedding_function(text) for text in texts]
        return cls.__from(
            texts,
            embeddings,
            embedding_function,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents asynchronously.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                faiss = await FAISS.afrom_texts(texts, embeddings)
        """
        embeddings = await embedding.aembed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Any,
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = zip(texts, text_embeddings)
                faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts, embeddings = zip(*text_embeddings)
        return cls.__from(
            list(texts),
            list(embeddings),
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_embeddings(
        cls,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Any,
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents asynchronously."""
        return cls.from_embeddings(
            text_embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Any,
        index_name: str = "index",
        **kwargs: Any,
    ) -> FAISS:
        """Load FAISS index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
            asynchronous: whether to use async version or not
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / f"{index_name}.faiss"))

        # load docstore and index_to_docstore_id
        with open(path / f"{index_name}.pkl", "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding_function: Any,
        **kwargs: Any,
    ):
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding_function, metadatas=metadatas, **kwargs)

    @classmethod
    def deserialize_from_bytes(
        cls,
        serialized: bytes,
        embeddings: Any,
        **kwargs: Any,
    ) -> FAISS:
        """Deserialize FAISS index, docstore, and index_to_docstore_id from bytes."""
        index, docstore, index_to_docstore_id = pickle.loads(serialized)
        return cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)

    def __add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        faiss = dependable_faiss_import()

        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )

        _len_check_if_sized(texts, metadatas, "texts", "metadatas")
        _metadatas = metadatas or ({} for _ in texts)
        documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)]

        _len_check_if_sized(documents, embeddings, "documents", "embeddings")
        _len_check_if_sized(documents, ids, "documents", "ids")

        # Add to the index.
        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        self.index.add(vector)

        # Add information to docstore and index.
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        self.index_to_docstore_id.update(index_to_id)
        return ids
