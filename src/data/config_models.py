from pydantic import BaseModel, Field, field_validator

from src.utils import get_active_systems

class IngestionConfig(BaseModel):
    # The ellipses '...' mean the field is strictly required
    source: str = Field(...,
                        description="The web URL or local path to the resource")
    game_name: str = Field(...,
                           description="The parent game name for folder routing")
    file_name: str = Field(..., description="The specific document name")

    # This validator runs automatically when the object is created
    @field_validator('game_name', 'file_name')
    @classmethod
    def normalize_strings(cls, value: str) -> str:
        """Automatically sanitizes the names for OS pathing."""
        return value.strip().lower().replace(" ", "_")

    