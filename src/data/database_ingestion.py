from typing import List
from pathlib import Path
import langchain_core
from langchain_chroma import Chroma


from src.utils import register_active_system
from src.env import PersistentVars

class IngestionDB:
    """
    Class to handle database ingestion operations efficiently by maintaining 
    a single active ChromaDB connection.
    """
    def __init__(self, db_path: Path, embedding_model):
        self.db_path = db_path
        self.embedding_model = embedding_model
        
        print(f"Connecting to persistent ChromaDB at {self.db_path}...")
        self.vector_db = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=self.embedding_model,
            collection_name=PersistentVars.COLLECTION
        )

    def add_to_vector_store(self, final_chunks: List[langchain_core.documents.base.Document], game_name: str):
        """
        Add data from semantic chunks to the persistent vector store.
        """
        # Add System Metadata Tag    
        for chunk in final_chunks:
            chunk.metadata["system"] = game_name.lower()

        # Append the new documents to the store
        print(f"Adding {len(final_chunks)} chunks to the collection for {game_name}...")
        self.vector_db.add_documents(documents=final_chunks)

        print(f"Ingestion complete! Database saved to: {self.db_path}")
        register_active_system(game_name, self.db_path.parent)