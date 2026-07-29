import os
from pathlib import Path
from typing import List, Dict
# from docling.document_converter import DocumentConverter


# Import helper classes
from src.data.manifest_manager import ManifestManager

# Importing existing pipeline components
from src.utils import get_game_files, get_semantic_chunks
from src.data.batch_parser import BatchDoclingParser
from src.data.database_ingestion import IngestionDB
              
                
class DataPipelineManager:
    """
    Class to handle entire Data to DB Pipeline.
    """
    def __init__(self, db_path: Path, embedding_model):
        self.db_path = db_path
        self.manifest = ManifestManager()
        
        # Initialize the database connection ONCE
        self.db = IngestionDB(db_path=db_path, embedding_model=embedding_model)

    def get_game_folders(self, paths: List[str]) -> Dict[str, List[str]]:
        folder_map = {}
        for i in paths:
            name = Path(i).parent.name
            if name in folder_map:
                folder_map[name].append(i)
            else:
                folder_map[name] = [i]
        return folder_map

    def discover_unprocessed_files(self) -> Dict[str, List[str]]:
        all_files = get_game_files()
        all_games = self.get_game_folders(all_files)
        
        unprocessed_games = {}
        for game_name, file_paths in all_games.items():
            unprocessed_paths = []
            for path in file_paths:
                file_identifier = f"{game_name}/{Path(path).name}"
                
                if not self.manifest.is_processed(file_identifier):
                    unprocessed_paths.append(path)
                else:
                    print(f"[SKIP] {file_identifier} already processed.")
            
            if unprocessed_paths:
                unprocessed_games[game_name] = unprocessed_paths
                
        return unprocessed_games

    def run_pipeline(self):
        os.makedirs(self.db_path, exist_ok=True)
        
        pending_data = self.discover_unprocessed_files()
        
        if not pending_data:
            print("Pipeline complete: No new files to process.")
            return

        for game_name, file_paths in pending_data.items():
            print(f"\n--- Processing {len(file_paths)} new file(s) for system: {game_name} ---")
            
            try:
                # 1. Batch Parse to Markdown
                batch_processor = BatchDoclingParser(chunk_size=5, hold_size=5)
                path_to_md = batch_processor.parse_to_markdown([Path(i) for i in file_paths])
                
                # 2. Semantic Chunking
                final_chunks = get_semantic_chunks(path_to_md)
                
                # 3. Add to Vector Store using the persistent connection
                self.db.add_to_vector_store(final_chunks=final_chunks, game_name=game_name)
                
                # 4. Mark files as processed in the manifest
                for path in file_paths:
                    file_identifier = f"{game_name}/{Path(path).name}"
                    self.manifest.mark_as_processed(file_identifier)
                    
                print(f"[SUCCESS] Finished ingesting {game_name}.")
                
            except Exception as e:
                print(f"[ERROR] Failed during ingestion for {game_name}: {e}")