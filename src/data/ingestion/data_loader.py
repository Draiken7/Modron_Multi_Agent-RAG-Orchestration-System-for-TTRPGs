class BaseLoader:
    """Base class interface for data loaders."""

    def __init__(self, config):
        self.config = config

    def load_data(self):
        """Method to load data. Should be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses should implement the load_data method.")


class WebURLLoader(BaseLoader):
    """Loader class for loading data by parsing it from a web URL."""

    def load_data(self):
        # Implement logic to load data from a web URL
        pass


class LocalFileLoader(BaseLoader):
    """Loader class for loading data from a uploaded local file."""

    def load_data(self):
        # Implement logic to load data from a local file
        pass


class GoogleDriveLoader(BaseLoader):
    """Loader class for loading data from Google Drive link."""

    def load_data(self):
        # Implement logic to load data from Google Drive
        pass
