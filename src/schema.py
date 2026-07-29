from enum import Enum
from pydantic import BaseModel, Field


from src.env import PersistentVars
from src.utils import get_active_systems


class DatabaseError(Exception):
    """Raises a problem with the database"""
    pass

# Assuming get_active_systems and PersistentVars are defined above
ACTIVE_SYSTEMS = get_active_systems(PersistentVars.SYSTEMS_MANIFEST)
if len(ACTIVE_SYSTEMS) == 0:
    raise DatabaseError("Database is empty of any active system rules!")

ACTIVE_SYSTEMS = ACTIVE_SYSTEMS["active_systems"]

SystemChoices = Enum(
    "SystemChoices", 
    {name: name for name in ACTIVE_SYSTEMS + ["multiple"]}
)

SYSTEMS_STRINGS = ", ".join(ACTIVE_SYSTEMS)

class QueryIntent(BaseModel):
    # CHANGED: Wrapped SystemChoices in a list to support multiple selections
    system: list[SystemChoices] = Field(
        description=(
            f"List of target TTRPG systems. Must strictly contain elements from [{SYSTEMS_STRINGS}]. "
            "If the user asks for a comparison, include all named systems. "
            "If unstated, ambiguous, or no system is established in memory, MUST be ['multiple']."
        )
    )
    clean_query: str = Field(
        description="The mechanical query rewritten into dense, high-signal search keywords."
    )


class RouterGrade(BaseModel):
    binary_score: str = Field(
        description="Is the router's classification and clean_query accurate and optimal? Answer strictly 'yes' or 'no'."
    )
    feedback: str = Field(
        description="If 'no', provide a brief 1-sentence reason explaining what was missed (e.g., 'Failed to capture Pathfinder from current query'). If 'yes', leave blank."
    )


class HallucinationGrade(BaseModel):
    binary_score: str = Field(
        description="Are all claims in the generation perfectly grounded in the provided facts? Answer strictly 'yes' or 'no'."
    )
    

class RouterGraderOutput(BaseModel):
    """Output schema for the Phase 1 Router Critic."""
    passed: bool = Field(
        description="Is the router's classification and clean_query accurate and optimal? Answer strictly 'yes' or 'no'. If yes, set this field to True else set it to False."
    )
    feedback: str = Field(
        description="If passed is False, provide a concise explanation of what the router got wrong and how to correct it. If True, leave empty."
    )
    
    
class GMGraderOutput(BaseModel):
    """Output Schema for the GM Critic"""
    passed: bool = Field(
        description="Are all claims in the generation perfectly grounded in the provided facts? Answer strictly True if yes or False otherwise."
    )
    feedback: str = Field(
        description="If passed is False, provide a concise explanation of what the game master model got wrong and how to correct it. If True, leave empty."
    )
    
    
class SystemCAOutput(BaseModel):
    """Output schema for the System CHecking and Augmenting model"""
    system: list[SystemChoices] = Field(
            description=(
                f"List of target TTRPG systems. Must strictly contain elements from [{SYSTEMS_STRINGS}]. "
                "If the user asks for a comparison, include all named systems. "
                "If unstated, ambiguous, or no system is established in memory, MUST be ['multiple']."
            )
        )
   
    
class KeyExpansionOutput(BaseModel):
    """Output schema for the Key word expansion model"""
    keywords: list[str] = Field(
        description="A list of 5 to 10 highly relevant keywords, TTRPG rule mechanics, pertinent to the user query and target systems."
    )
    
    
class HyDEOutput(BaseModel):
    """Output schema for the HyDE model"""
    keywords: list[str] = Field(
        description="A dense list of unique, high-value mechanical jargon, exact rule phrases, and keywords extracted from the hallucinated answer."
    )
    system: list[str] = Field(
        description="The specific TTRPG system(s) the rules belong to (use full game names connected by underscores)."
    )