import re
import os
from typing import List, Dict
from pathlib import Path
import json

import langchain_core
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.env import PersistentVars

def find_scattered_keywords(keyword: str, text_list: list[str]) -> list[str]:
    """
    # --- Example Usage ---
        my_keyword = "elden"
        my_strings = [
            "I love playing elden ring",           # Perfect match
            "e_l_d_e_n_ring_save_file",          # Underscores added
            "e.x.l.x.d.x.e.x.n",                 # Random chars inserted
            "The elder scrolls is good",         # Fails (missing the 'n' in order)
            "e l   d   e n",                     # Spaces added
            "totally unrelated string"           # Fails
        ]

        results = find_scattered_keywords(my_keyword, my_strings)
        for res in results:
            print(f"Matched: {res}")
    """
    
    # 1. Escape the characters (in case your keyword contains regex symbols like '.')
    # 2. Join them with the non-greedy wildcard '.*?'
    regex_string = ".*?".join(re.escape(char) for char in keyword)
    
    # 3. Compile the regex (IGNORECASE makes it case-insensitive)
    pattern = re.compile(regex_string, re.IGNORECASE)
    
    # 4. Filter the list
    matched_strings = [text for text in text_list if pattern.search(text)]
    return matched_strings


def get_game_files() -> List[Path]:
    """
    Returns a list of paths to all game files.
    """   
    # 1. Dynamically locate the 'data' folder in the same directory as this script
    script_dir = os.getcwd()
    data_folder = os.path.join(script_dir, PersistentVars.DATA_FOLDER)

    all_files = []

    # 2. Walk through the data folder
    for root, dirs, files in os.walk(data_folder):
        # 3. Combine the root path with the file name to get the absolute path
        pdf_files = [
            os.path.join(root, file) 
            for file in files 
            if file.lower().endswith(".pdf")
        ]
        all_files.extend(pdf_files)
        
    print(f"Found {len(all_files)} game files!")
    print("The Files are:")

    # 4. Removed the extra "\n" since print() adds one automatically
    for file_path in all_files:
        print(file_path)
        
    return all_files


def register_active_system(game_name: str, path: Path):
    """
    Add active system to the json for faster lookup.
    """
    # Add System tag to a systems.json for faster lookup        
    systems_file = path / "systems.json"
    target_system = game_name.lower()

    # 1. Safely load existing data
    try:
        with open(systems_file, 'r') as f:
            file_data = json.load(f)
            
            # SAFEGUARD: If the loaded data is the old list format, overwrite it
            if not isinstance(file_data, dict):
                file_data = {"active_systems": []}
                
            # Ensure the key exists just in case
            if "active_systems" not in file_data:
                file_data["active_systems"] = []
                
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is completely empty, start with the correct dictionary structure
        file_data = {"active_systems": []}

    # 2. Append the new data to the list (checking for duplicates first)
    if target_system not in file_data["active_systems"]:
        file_data["active_systems"].append(target_system)

    # 3. Write the data back to the file
    with open(systems_file, 'w') as f:
        json.dump(file_data, f, indent=4)
        
            
def get_active_systems(path: Path) -> Dict:
    """Returns all the active systems in the database."""
    # Add System tag to a systems.json for faster lookup        
    # systems_file = path / "systems.json"
    systems_file = path

    try:
        with open(systems_file, 'r') as f:
            file_data = json.load(f)
            
            # SAFEGUARD: If the loaded data is the old list format, overwrite it
            if (not isinstance(file_data, dict)) or ("active_systems" not in file_data):
                return {}
            
            return file_data
                
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_semantic_chunks(master_md_path: Path) -> List[langchain_core.documents.base.Document]:
    """
    Returns a list of semantic chunks from the master markdown file.
    """
    print("Loading master Markdown file...")
    with open(master_md_path, "r", encoding="utf-8") as f:
        markdown_document = f.read()

    # 1. Define the Semantic Hierarchy
    # This tells LangChain to look for H1, H2, and H3 tags to create logical breaks.
    headers_to_split_on = [
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Subsection"),
    ]

    # 2. Perform the Semantic Split
    print("Splitting text by Markdown headers...")
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep the headers in the text so the LLM can read them
    )
    semantic_chunks = markdown_splitter.split_text(markdown_document)
    print(f"Generated {len(semantic_chunks)} semantic chunks.")

    # 3. The Safety Net (Recursive Character Splitter)
    # Some TTRPG sections (like a massive spell list) might still be too long for an embedding model.
    # This secondary splitter ensures no single chunk exceeds your token limits, while preserving the metadata.
    print("Applying strict length constraints...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   # Maximum characters per chunk
        chunk_overlap=200  # Overlap to maintain context between split paragraphs
    )

    final_chunks = text_splitter.split_documents(semantic_chunks)
    print(f"Final RAG-ready chunk count: {len(final_chunks)}")

    # --- Inspect the results ---
    print("\n--- SAMPLE CHUNK ---")
    sample = final_chunks[10] # Grab an arbitrary chunk to inspect
    print(f"Metadata: {sample.metadata}")
    print(f"Content:\n{sample.page_content[:200]}...")
    return final_chunks


def get_formatted_list(input: list[str]) -> str:
    """
    Return a string made from a list of strings.
    """
    if input:
        # Joins the list items with a newline so it reads like a real script
        return "\n".join(input)
    else:
        return "None"