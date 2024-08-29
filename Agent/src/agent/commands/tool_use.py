from typing import Any

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores.utils import DistanceStrategy

from agent.commands.core import UseTool
from agent.memory import MemKey
from agent.tools.vector_db.faiss import FAISS_DB


def initialise_vector_db(
    db_type: str | None,
    path_to_docs: str | None,
    path_to_database: str | None,
    database_name: str | None,
    embedding_function,
    distance_function: DistanceStrategy,
    path_to_save_db: str | None,
    name_to_save_db: str | None,
):
    if path_to_database and database_name:
        # load database that has been already initialised.
        if db_type == "faiss":
            db = FAISS_DB.load_local(
                path_to_database, embedding_function, database_name, distance_strategy=distance_function
            )
        else:
            raise ValueError("The provided db_type is not supported by `agent`. Please provide a supported type")
    elif path_to_docs:
        loader = DirectoryLoader(path_to_docs)
        documents = loader.load()
        if db_type == "faiss":
            db = FAISS_DB.from_documents(
                documents=documents, embedding_function=embedding_function, distance_strategy=distance_function
            )
        else:
            raise ValueError("The provided db_type is not supported by `agent`. Please provide a supported type")
        if path_to_save_db:
            db.save_local(path_to_save_db, name_to_save_db)
    else:
        raise ValueError("Either path and name to a saved database should be passed or a path to docs")
    return db


class VectorDB(UseTool):
    """Initialise and query a vector db.

    Input:
        db_type: faiss - Type of the vectorstore that will be initialised.
        For now only faiss is supported. Need to add more types

        path_to_database: If this is not None, a database_name should also be provided.
        This path will be used to load the database from a local save.

        database_name: Name of the database that will be loaded from the local file

        path_to_docs: Path to a folder that contains files that will be used to initialise a database.

        distance_function: The function that will be used to compute the embedding distances

        path_to_save_db: If this is provided a database initialised from documents will be saved to this path


    Output:
        memory_key: MemKey.DB_QUERY - Stores the query that is used to query the database
        memory_key: MemKey.RAG_RETRIEVAL - stores all retrieved text from the database
    """

    name: str = "vector_db_query"
    description: str = "Generate using an LLM a query and a vector database to find the closest matching answer"

    # Required attributes from Command()
    required_prompt_templates: dict[str, str] = {"ask_template": "query.jinja"}
    output_keys: dict[str, MemKey] = {
        "rag_retrieval_mem_key": MemKey.RAG_RETRIEVAL,
        "db_query_mem_key": MemKey.DB_QUERY,
    }

    # VectorDB specific attributes
    db_type: str = "faiss"
    query_template: str | None = None
    path_to_database: str | None = None
    database_name: str | None = None
    path_to_docs: str | None = None
    distance_function: DistanceStrategy = DistanceStrategy.COSINE
    path_to_save_db: str | None = None
    name_to_save_db: str | None = "index"
    vector_db: Any = None
    top_k: int = 4

    def func(self, agent, rag_retrieval_mem_key, db_query_mem_key, ask_template):
        if self.vector_db is None:
            self.vector_db = initialise_vector_db(
                self.db_type,
                self.path_to_docs,
                self.path_to_database,
                self.database_name,
                agent.embedding.embed_text,
                self.distance_function,
                self.path_to_save_db,
                self.name_to_save_db,
            )
        query_prompt = agent.prompt_builder([ask_template], {"memory": agent.memory})
        db_query = agent.llm.chat_completion(query_prompt, lambda x: x)
        db_query = db_query.split("Query:")[-1]

        agent.memory.store(db_query, {db_query_mem_key})
        returned_docs = self.vector_db.similarity_search(db_query, k=self.top_k)
        for doc in returned_docs:
            agent.memory.store(doc.page_content, {rag_retrieval_mem_key})


class GoogleSearch(UseTool):
    """Performs google search to find related urls that contain an answer related to the query.

    Input:
        search_engine: The search engine that will be used.
        top_k: The number of returned results

    Output:
        memory_key: MemKey.LINK_RETRIEVED - link of the retrieved website
        memory_key: MemKey.TITLE_RETRIEVED - title of the retrieved website
        memory_key: MemKey.SNIPPET_RETRIEVED - snippet from the retrieved website
        memory_key: MemKey.GOOGLE_QUERY - stores the query that is used to query google
    """

    search_engine: Any
    top_k: int = 1

    required_prompt_templates: dict[str, str] = {"ask_template": "query.jinja"}
    output_keys: dict[str, MemKey] = {
        "link_retrieved_key": MemKey.LINK_RETRIEVED,
        "title_retrieved_key": MemKey.TITLE_RETRIEVED,
        "snippet_retrieved_key": MemKey.SNIPPET_RETRIEVED,
        "google_query_mem_key": MemKey.GOOGLE_QUERY,
    }

    def func(
        self,
        agent,
        link_retrieved_key,
        snippet_retrieved_key,
        title_retrieved_key,
        google_query_mem_key,
        ask_template,
    ):
        query_prompt = agent.prompt_builder([ask_template], {"memory": agent.memory})
        google_query = agent.llm.chat_completion(query_prompt, lambda x: x)
        google_query = google_query.split("Query:")[-1]

        agent.memory.store(google_query, {google_query_mem_key})

        results = self.search_engine.results(google_query, self.top_k)
        for res in results:
            agent.memory.store(res["title"], {title_retrieved_key})
            agent.memory.store(res["snippet"], {snippet_retrieved_key})
            agent.memory.store(res["link"], {link_retrieved_key})


class URLRetrieval(UseTool):
    search_engine: Any
    top_k: int = 1  # must be the same that GoogleSearch uses
    chunk_size: int = 5000
    chunk_overlap: int = 200

    text_splitter: Any = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    required_prompt_templates: dict[str, str] = {}
    input_keys: dict[str, str] = {"link_retrieved_key": MemKey.LINK_RETRIEVED}
    output_keys: dict[str, str] = {"text_retrieved_key": MemKey.TEXT_RETRIEVED, "num_of_chunks_key": MemKey.NUM_CHUNKS}

    def func(self, agent, link_retrieved_key, text_retrieved_key, num_of_chunks_key):
        urls_from_mem = agent.memory.retrieve_all({link_retrieved_key: 1.0})[: self.top_k]
        urls = []
        for url in urls_from_mem:
            urls.append(url.content)
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        num_of_chunks = 0
        for doc in docs:
            text = doc.page_content
            splitted_text = self.text_splitter.split_text(text)
            for t in splitted_text:
                num_of_chunks += 1
                agent.memory.store(t, {text_retrieved_key})

        agent.memory.store(num_of_chunks, {num_of_chunks_key})
