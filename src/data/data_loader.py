import shutil
import os
from pathlib import Path
import requests


from src.data.config_models import IngestionConfig
from src.utils import find_scattered_keywords
from src.env import PersistentVars


class NotImplementedError(Exception):
    """Custom exception for unimplemented features."""
    pass


class BaseLoader:
    """Base class interface for data loaders."""

    def __init__(self, IngestionConfig):
        # let config contain the source type and any other relevant parameters
        self.config = IngestionConfig

        # must have a game name for folder and downloaded file name

        # requires webURL or check if null
        # requires TTRPG name for folder and downloaed file name

        # Reuires file path for local file loader
        # can have alternate file name for local file loader

    def load_data(self):
        """Method to load data. Should be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses should implement the load_data method!")


class WebURLLoader(BaseLoader):
    """Loader class for loading data by parsing it from a web URL."""

    def load_data(self):
        url = self.config.source

        game = self.config.game_name

        ttrpg_name = self.config.file_name

        # 1. Calculate paths BEFORE downloading
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[0]

        # This automatically routes to the specific game's subfolder
        save_dir = project_root/  PersistentVars.DATA_FOLDER / game
        save_path = save_dir / f"{ttrpg_name}.pdf"

        # 2. Check for duplicates to prevent overwriting
        if save_path.exists():
            raise FileExistsError(
                f"The file {save_path.name} already exists in {save_dir.name}.")

        # 3. Proceed with the download since the file does not exist
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return save_path


class LocalFileLoader(BaseLoader):
    """Loader class for loading data from a uploaded local file."""

    def load_data(self):
        file_path = self.config.source

        source_file = Path(file_path)
        if not source_file.exists():
            raise FileNotFoundError(
                f"The specified file does not exist: {file_path}")

        name = self.config.file_name
        folder_name = self.config.game_name

        current_file = Path(__file__).resolve()
        project_root = current_file.parents[0]

        # Route to the specific subfolder
        save_dir = project_root / PersistentVars.DATA_FOLDER / folder_name
        save_path = save_dir / \
            f"{name if name is not None else source_file.name}.pdf"

        # Check for duplicates before copying
        if save_path.exists():
            raise FileExistsError(
                f"The file {save_path.name} already exists in {save_dir.name}.")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Recommendation: Use shutil.copy2 to preserve the original file's metadata
        shutil.copy2(source_file, save_path)

        return save_path


class GoogleDriveLoader(BaseLoader):
    """Loader class for loading data from Google Drive link."""
    # not implemented yet


class Ingestion:
    """Class to handle data ingestion from various sources."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        
        self.source_uri = self.config.source
        
        # 1. Fix typo in config
        game = self.config.game_name

        # 2. OPTIMIZATION: Get only top-level directory names efficiently
        try:
            games = [d.name for d in os.scandir(PersistentVars.DATA_FOLDER) if d.is_dir()]
        except FileNotFoundError:
            games = [] # Handle case where 'data' folder doesn't exist yet

        # 3. Use the regex function from earlier
        matched = find_scattered_keywords(game, games)
        complete_match = game in games

        # 4. Simplify condition (matched is a list, so 'if matched' checks if it has items)
        if matched and not complete_match:
            # 5. Fix string formatting for the print statement
            matched_names = "\n - ".join(matched)
            print(f"Game name must be unique for a new game or match existing ones!\nCurrent name is similar to:\n - {matched_names}")
            
            ans = input("\nDo you wish to continue with the given game name nonetheless? [y/n] ")
            
            if ans.lower() != 'y':
                # 6. Fix the typo 'joing' and string formatting in the ValueError
                all_games = "\n - ".join(games)
                raise ValueError(f"Choose a completely unique name, or match one of the following exactly:\n - {all_games}")

        self.loader = self._auto_detect_loader()

    def _auto_detect_loader(self):
        """Intelligently routes to the correct strategy based on string analysis."""

        if "drive.google.com" in self.source_uri:
            raise NotImplementedError(
                "Google Drive loader is not implemented yet.")
            # return GoogleDriveLoader(self.config)

        elif self.source_uri.startswith(("http://", "https://")):
            return WebURLLoader(self.config)

        else:
            return LocalFileLoader(self.config)
