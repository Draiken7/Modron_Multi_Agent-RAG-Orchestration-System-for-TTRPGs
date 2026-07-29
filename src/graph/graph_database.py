from enum import Enum
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


from src.env import PersistentVars

# Strategy Enum to control search engine execution
class RetrievalStrategy(Enum):
    HYBRID = "hybrid"           # Vector + BM25 Ensemble (Standard & Keyword Expansion)
    VECTOR_ONLY = "vector_only" # Pure Cosine Similarity (HyDE paragraphs)
    

class DatabaseOP:
    """
    Unified Database Operator designed for LangGraph multi-agent retrieval.
    Loaded ONCE as a global application resource.
    """
    def __init__(self):
        # Opening custom log File to write to
        log_file = open(PersistentVars.LOG_FILE, "a", encoding="utf-8")
        
        print("[SYSTEM] Booting Database Operator & Loading Embeddings into GPU...", file=log_file)
        
        self.db_path = PersistentVars.DB_PATH
        self.collection = PersistentVars.COLLECTION
        
        # Initialize GPU Embedding Model ONCE
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=PersistentVars.EMBBEDDING_MODEL,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Intialize the Persistent Vector Database Connection
        self.vector_db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model,
            collection_name=self.collection
        )
        
        # Cache dictionary to store BM25 instances in RAM once initialized
        # Format: {"dnd5e": BM25Retriever_Instance, "pathfinder": BM25Retriever_Instance}
        self._bm25_cache = {}       
        
        print("[SYSTEM] Database Operator Ready.", file=log_file)
        
        log_file.close()
        
    def _get_or_create_bm25(self, system: str, k: int) -> BM25Retriever:
        """
        Private helper method: Lazy-loads and caches BM25 indexes in RAM.
        Prevents rebuilding the index from Chroma on every single user query.
        """
        with open(PersistentVars.LOG_FILE, 'a', encoding="utf-8") as log_file:
            if system not in self._bm25_cache:
                print(f"  └─► [BM25 CACHE MISS] Building in-memory BM25 index for system: '{system}'...", file=log_file)
                
                # Fetch raw text documents for this system ONCE
                system_docs_data = self.vector_db.get(
                    where={PersistentVars.METADATA["systems"]: system}
                )
                
                # Reconstruct LangChain Document objects
                system_documents = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(system_docs_data["documents"], system_docs_data["metadatas"])
                ]
                
                # Create and store the BM25 index in our cache dictionary
                bm25_retriever = BM25Retriever.from_documents(system_documents)
                self._bm25_cache[system] = bm25_retriever
                
        # Retrieve cached instance and dynamically update top_k
        retriever = self._bm25_cache[system]
        retriever.k = k
        return retriever
    
    def retrieve(self, search_queries: str | list[str], target_systems: str | list[str], strategy: RetrievalStrategy = RetrievalStrategy.HYBRID, k: int = 4) -> list[Document]:
        """
        The universal entry point for ALL LangGraph retrieval nodes.
        Handles query fusion, strategy routing, and multi-system searches.
        """
        # 1. Normalize Inputs
        if isinstance(target_systems, str):
            target_systems = [target_systems]
            
        if isinstance(search_queries, list):
            # Fuse list of strings (e.g., clean_query + expanded_keywords) into one query space
            fused_query = " ".join(search_queries)
        else:
            fused_query = search_queries
            
        all_retrieved_docs = []
        
        # 2. Iterate through each targeted game system
        for system in target_systems:
            
            # --- STRATEGY A: HYBRID SEARCH (Vector + Cached BM25) ---
            if strategy == RetrievalStrategy.HYBRID:
                # Engine 1: Vector Retriever
                vector_retriever = self.vector_db.as_retriever(
                    search_kwargs={"k": k, "filter": {PersistentVars.METADATA["systems"]: system}}
                )
                
                # Engine 2: Cached BM25 Retriever
                bm25_retriever = self._get_or_create_bm25(system, k=k)
                
                # Fusion via Reciprocal Rank Fusion (RRF)
                ensemble = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                
                system_docs = ensemble.invoke(fused_query)

            # --- STRATEGY B: VECTOR-ONLY SEARCH (For HyDE Engine) ---
            elif strategy == RetrievalStrategy.VECTOR_ONLY:
                # Pure semantic search (skips BM25 keyword matching)
                vector_retriever = self.vector_db.as_retriever(
                    search_kwargs={"k": k, "filter": {PersistentVars.METADATA["systems"]: system}}
                )
                system_docs = vector_retriever.invoke(fused_query)

            # Append results for this system
            all_retrieved_docs.extend(system_docs)

        return all_retrieved_docs
    
    def get_unfiltered_top_k(self, query: str, active_systems: list[str]) -> list[Document]:
        """Fetches top 1 matching chunk from EVERY active system for Path B Disambiguation."""
        
        disambiguation_docs = []
        for system_key in active_systems:
            retriever = self.vector_db.as_retriever(
                search_kwargs={
                    "k": 1, 
                    "filter": {PersistentVars.METADATA["systems"]: system_key}
                }
            )
            system_docs = retriever.invoke(query) # Broad pull across system
            if system_docs:
                disambiguation_docs.extend(system_docs)
        return disambiguation_docs
    
    def format_docs_with_metadata(self, docs: list[Document]) -> str:
        """Formats retrieved chunks with system tags for LLM context injection."""
        formatted_blocks = []
        for doc in docs:
            system_tag = doc.metadata.get(PersistentVars.METADATA["systems"], "General/Unknown").upper()
            formatted_blocks.append(f"--- [SYSTEM: {system_tag}] ---\n{doc.page_content}")
        return "\n\n".join(formatted_blocks)