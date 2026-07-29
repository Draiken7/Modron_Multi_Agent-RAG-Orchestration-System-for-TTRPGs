import json
import os


class ManifestManager:
    def __init__(self, manifest_path="processed_files.json"):
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> set:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: {self.manifest_path} is corrupted. Starting fresh.")
                return set()
        return set()

    def _save_manifest(self):
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(self.manifest)), f, indent=4)

    def is_processed(self, file_identifier: str) -> bool:
        return file_identifier in self.manifest

    def mark_as_processed(self, file_identifier: str):
        if file_identifier not in self.manifest:
            self.manifest.add(file_identifier)
            self._save_manifest()