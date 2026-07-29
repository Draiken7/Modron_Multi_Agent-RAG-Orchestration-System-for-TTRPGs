from langgraph.graph import END

from src.graph.graph_state import GameMasterState
from src.env import Nodes



def route_phase_1(state: GameMasterState) -> str:
    """
    Conditional Edge to choose action after Route 1. Either move on to retrieval stage and phase 2, else the Fail safe or retry.
    """
    if state.phase_1_passed:
        return Nodes.RETRIEVER
    elif state.phase_1_retries > 2:
        return Nodes.RFAIL
    else:
        return Nodes.ROUTER
        
def route_retrieval(state: GameMasterState) -> str:
    """
    Route to retriever node.
    """
    systems = state.guessed_systems
    if not systems or 'multiple' in systems:
        return Nodes.SUMMARY
    else:
        # no phase 2 yet
        return Nodes.GAMEMASTER
    
def route_phase_2(state: GameMasterState) -> str:
    """
    Conditional Edge to choose action after GM generates response andit is audited by the gmcritic.
    """
    if state.phase_2_passed:
        return Nodes.MEMORY
    elif state.phase_2_retries == 2:
        return Nodes.SYSTEMCA # and keyword Expansion
    elif state.phase_2_retries == 3:
        return Nodes.HYDE
    elif state.phase_2_retries == 4:
        return Nodes.GMFAIL
    else:
        return Nodes.GAMEMASTER