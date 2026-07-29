from langchain_core.runnables.base import RunnableSequence

from src.graph.graph_state import GameMasterState
from src.graph.graph_database import DatabaseOP, RetrievalStrategy
from src.env import PersistentVars
from src.utils import get_formatted_list



class RouterNode:
    """Node that Analyzes the user's query to identify the game system and clean the text.
        Acts as the first step in Phase 1. """
    def __init__(self, chain: RunnableSequence, active_systems: list[str]):
        self.chain = chain
        self.active_systems = active_systems
    
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print(f"\n  └─► [NODE: ROUTER] Routing query (Attempt {state.phase_1_retries + 1})...", file=log_file)
            
            formatted_active_systems = ", ".join(self.active_systems)
            
            formatted_history = get_formatted_list(state.recent_history)
            
            # Read from the Pydantic State using dot-notation
            prompt_inputs = {
                "question": state.user_query,
                "chat_summary": state.chat_summary,
                "recent_history": formatted_history,
                # We pass the feedback in. If it's attempt #1, this is an empty string.
                # If it's attempt #2, it contains the Critic's correction!
                "feedback": state.phase_1_feedback,
                "active_systems": formatted_active_systems,
                # "format_instructions": self.chain.last.get_format_instructions()
            }
            
            try:
                # Invoke our pre-built LLM chain
                # (Assuming router_chain is configured to output our Stage 1 JSON schema)
                intent: self.chain.output_schema = self.chain.invoke(prompt_inputs)
                
                sys = [system.name for system in intent.system] if intent.system else ['multiple']
                
                # Return the partial dictionary update
                return {
                    "guessed_systems": sys,
                    "clean_query": intent.clean_query,
                    "phase_1_retries": state.phase_1_retries + 1,
                    # Reset feedback on a new attempt so old feedback doesn't linger
                    "phase_1_feedback": ""
                }
                
            except Exception as e:
                # Graceful Error Handling (e.g., the 8B model hallucinated bad JSON)
                print(f"      ↳ [FAIL] Router LLM format error: {e}", file=log_file)
                return {
                    "phase_1_passed": False, 
                    "phase_1_feedback": f"System crash during routing: {str(e)}. Please try again and ensure strict JSON output.",
                    "phase_1_retries": state.phase_1_retries + 1
                }
                
                
class RouterCriticNode:
    """
    Analyzes the Router's response to check for proper systems and filler removal.
    """
    def __init__(self, chain: RunnableSequence, active_systems: list[str]):
        self.chain = chain
        self.active_systems = active_systems
    
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print(f"\n  └─► [NODE: PHASE 1 CRITIC] Auditing proposed intent...", file=log_file)

            # Format list into a clean, readable string for the prompt
            formatted_active_systems = ", ".join(self.active_systems)
            
            formatted_history = get_formatted_list(state.recent_history)
                
            # Format the guessed systems to a string as well
            guessed = ", ".join(state.guessed_systems)

            # 1. Package state data for the Grader Prompt
            prompt_inputs = {
                "question": state.user_query,
                "chat_summary": state.chat_summary if state.chat_summary else "None",
                "recent_history": formatted_history,
                "active_systems": formatted_active_systems,
                "selected_systems": guessed,
                "clean_query": state.clean_query,
                # "format_instructions": self.chain.last.get_format_instructions()
            }

            try:
                # 2. Invoke the Grader Chain
                grade = self.chain.invoke(prompt_inputs)

                if grade.passed:
                    print("      ↳ [CRITIC PASS] Proposed intent approved.", file=log_file)
                    return {
                        "phase_1_passed": True,
                        "phase_1_feedback": "" # Clear feedback on success
                    }
                else:
                    print(f"      ↳ [CRITIC FAIL] Feedback generated: '{grade.feedback}'", file=log_file)
                    return {
                        "phase_1_passed": False,
                        "phase_1_feedback": grade.feedback
                    }

            except Exception as e:
                # Catch LLM JSON/Parsing errors gracefully
                print(f"      ↳ [CRITIC ERROR] Parser failed during grading: {e}", file=log_file)
                return {
                    "phase_1_passed": False,
                    "phase_1_feedback": f"Critic execution failed due to format error: {str(e)}. Please re-verify intent output."
                }
                
    
class RetrieverNode:
    """
    Gets the user/clean query and db operation object to retrieve documents.
    """
    def __init__(self, active_systems: list[str], db: DatabaseOP):
        self.db = db
        self.active_systems = active_systems
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print("\n  └─► [NODE: RETRIEVAL] Executing Hybrid Search via DatabaseOP...", file=log_file)
            
            systems = state.guessed_systems
            clean_query = state.clean_query
            # Handle Fallback / Ambiguous Multi-System Routing
            if not systems or "multiple" in systems:
                # If no system was locked down, pull top chunks across all systems using un-filtered search
                # (Assuming you have an active systems list accessible in state or config)
                docs = self.db.get_unfiltered_top_k(state.user_query, self.active_systems)
                          
            else:
                # Execute Hybrid Search (Vector + BM25 Ensemble) using your class
                docs = self.db.retrieve(
                    search_queries=clean_query,
                    target_systems=systems,
                    strategy = RetrievalStrategy.HYBRID,
                    k = 4
                )
                
                # Format the retrieved documents using your built-in formatter method
            if not docs:
                formatted_context = (
                    "SYSTEM ALERT: A search was conducted, but no official rules "
                    "matching this mechanic were found within the specified database."
                )
                print("      ↳ [ALERT] No relevant documents found.", file=log_file)
            else:
                formatted_context = self.db.format_docs_with_metadata(docs)
                print(f"      ↳ [SUCCESS] Retrieved and formatted {len(docs)} chunks.", file=log_file)
                
            return {
                "retrieved_context": formatted_context
            }
     
   
class RouterFailSafeNode:
    """
     Handles state management in case of Router Failure. This sets the guessed system list to multiple overwriting anything else that was written to that variable.
    """
    def __init__(self):
        pass
    
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            systems_str = ", ".join(state.guessed_systems)
            print(f"\n  └─► [NODE: RouterFailSafe] Executing guessed system overwrite to 'multiple' for summarization (Previous: {systems_str})...", file=log_file)
            
            # Must return a dictionary for a langGraph Node
            return {
                "guessed_systems": ['multiple']
            }
   
            
class SummarizerNode:
    """
    Summarizes the documents provided for a graceful failure state.
    """
    def __init__(self, chain: RunnableSequence, active_systems: list[str]):
        self.chain = chain
        self.active_systems = active_systems
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print("\n  └─► [NODE: SUMMARIZER] Executing general summarization...", file=log_file)

            # Format list into a clean, readable string for the prompt
            formatted_active_systems = ", ".join(self.active_systems)
            
            formatted_history = get_formatted_list(state.recent_history)
                
            prompt_inputs = {
                "chat_summary": state.chat_summary if state.chat_summary else "None",
                "recent_history": formatted_history,
                "retrieved_context": state.retrieved_context if state.retrieved_context else "None",
                "question": state.user_query,
                "active_systems": formatted_active_systems
            }
            
            try:
                # invoke the summarizer chain
                summarizer = self.chain.invoke(prompt_inputs)
                
                return {
                    "draft_response": summarizer
                }
                
            except Exception as e:
                print(f"      ↳ [SUMMARIZER ERROR] Parser failed during summarization: {e}", file=log_file)
                return {
                    "draft_response": "Seems like I was unable to process the query properly. Can you please rephrase your question/prompt to help me out here?"
                }     


class MemoryNode:
    """
    Injects current chat into short term or long term memory
    """
    def __init__(self, chain: RunnableSequence, window: int = 10):
        self.chain = chain
        self.window = window
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, "a", encoding='utf-8') as log_file:
            print("\n  └─► [NODE: Memory] Executing Memory management for current conversation...", file=log_file)
            
            # Create a copy of the history to avoid mutating LangGraph state directly
            history = list(state.recent_history) if state.recent_history else []
            history.extend([
                f"User: {state.user_query}",
                f"AI: {state.draft_response}"
            ])
            
            if len(history) > self.window:
                oldest_query = history.pop(0)
                oldest_response = history.pop(0)
                
                # Extract .content from the Message objects
                new_lines = f"User: {oldest_query.content}\nAI: {oldest_response.content}\n"
                
                prompt_inputs = {
                    "chat_summary": state.chat_summary if state.chat_summary else "None",
                    "new_lines": new_lines
                }
                
                try:
                    mem = self.chain.invoke(prompt_inputs)
                    return {
                        "recent_history": history,
                        "chat_summary": mem
                    }
                
                except Exception as e:
                    print(f"      ↳ [MEMORY ERROR] Node chain failed!: {e}", file=log_file)
                    return {
                        "chat_summary": state.chat_summary if state.chat_summary else "None",
                        "recent_history": history # Ensure we still return the updated history on fail
                    }
                
            return {
                "recent_history": history
            }
      

class GameMasterNode:
    """
    The Generation for the RAG system that works on Retrieved Context and User query to Generate Answers.
    """
    def __init__(self, chain: RunnableSequence):
        self.chain = chain
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, "a", encoding='utf-8') as log_file:
            print("\n  └─► [NODE: GameMaster] Generating Response based on Retrieved context...", file=log_file)
            
            formatted_history = get_formatted_list(state.recent_history)
                
            prompt_inputs = {
                "chat_summary": state.chat_summary if state.chat_summary else "None",
                "recent_history": formatted_history,
                "retrieved_context": state.retrieved_context,
                "question": state.user_query
            }
            
            
            try:
                response = self.chain.invoke(prompt_inputs)
                
                return {
                    "draft_response": response,
                    "phase_2_retries": state.phase_2_retries + 1
                }
            except Exception as e:
                print(f"      ↳ [GAMEMASTER ERROR] Parser Failed to execute: {e}", file=log_file)
                return {
                    "draft_response": "I encountered an error and was not able to answer your question. Please try again and include more context to the questions, if possible.",
                    "phase_2_retries": state.phase_2_retries + 1
                }
                

class GMCriticNode:
    """
    Checks the Drafted GM response for hallucination by comparing with the context.
    """
    def __init__(self, chain: RunnableSequence):
        self.chain = chain
    
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print(f"\n  └─► [NODE: GM CRITIC] Auditing proposed response for hallucinations...", file=log_file)
            
            
            prompt_inputs = {
                "retrieved_context": state.retrieved_context,
                "draft_response": state.draft_response
            }
            
            try:
                grade = self.chain.invoke(prompt_inputs)
                
                if grade.passed:
                    print("      ↳ [GMCRITIC PASS] Proposed intent approved.", file=log_file)
                    return {
                        "phase_2_passed": True,
                        "phase_2_feedback": "" # Clear feedback on success
                    }
                else:
                    print(f"      ↳ [GMCRITIC FAIL] Feedback generated: '{grade.feedback}'", file=log_file)
                    return {
                        "phase_2_passed": False,
                        "phase_2_feedback": grade.feedback
                    }
            except Exception as e:
                print(f"      ↳ [CRITIC ERROR] Parser failed during grading: {e}", file=log_file)
                return {
                    "phase_2_passed": False,
                    "phase_2_feedback": f"GMCritic execution failed due to format error: {str(e)}. Please re-verify GM output."
                }
                

class GMFailSafeNode:
    """
    Overwrites the draft response with a safe apology if the GM fails the hallucination check repeatedly.
    """
    def __init__(self):
        pass
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print("\n  └─► [NODE: GMFAILSAFE] Max retries hit. Scrubbing hallucination...", file=log_file)
            
        return {
            "draft_response": "I apologize, but I am having trouble verifying the official rules for this mechanic in my database right now. Could you try rephrasing your question or providing more context?"
        }
 
        
class SystemCANode:
    """
    Checks if the system identified is correct or augments it with another best guess.
    """
    def __init__(self, chain: RunnableSequence, active_systems: list[str]):
        self.chain = chain
        self.active_systems = active_systems
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print("\n       └─► [NODE: SYSTEMCA] Checking and Augmenting Game Systems for better Retrieval...", file=log_file)
            
            guessed_systems = set(state.guessed_systems)
            
            # Format list into a clean, readable string for the prompt
            formatted_active_systems = ", ".join(self.active_systems)
            
            formatted_history = get_formatted_list(state.recent_history)
            
            prompt_inputs = {
                "active_systems": formatted_active_systems,
                "chat_summary": state.chat_summary,
                "recent_history": formatted_history,
                "question": state.user_query,
            }
            
            try:
                
                # call the chain with the prompt and returns the pydantic object
                system = self.chain.invoke(prompt_inputs)
                
                extracted_systems = [
                    s.value if hasattr(s, 'value') else str(s) 
                    for s in system.system
                ]
                
                if 'multiple' in extracted_systems:
                    guessed_systems = {'multiple'}
                else:
                    extracted_systems = set(extracted_systems)
                    # Check if guessed systems are already present, else add them
                    # nested will be n^2, set conversion might be 2n
                    guessed_systems |= extracted_systems
                
                return {
                    "guessed_systems": list(guessed_systems)
                }
            except Exception as e:
                print(f"\n       └─► [SYSTEMCA ERROR] Parsing failed: {e}...", file=log_file)
                return {
                    "guessed_systems": list(guessed_systems)
                }

                
class KeyExpansionNode:
    """
    Augments more keywords for a better query.
    """
    def __init__(self, chain: RunnableSequence):
        self.chain = chain
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print("\n       └─► [NODE: KEYEXPANSION] Generating a set of new keywords...", file=log_file)
            
            formatted_guessed_systems = get_formatted_list(state.guessed_systems)
            formatted_history = get_formatted_list(state.recent_history)
            formatted_keywords = get_formatted_list(state.expanded_keywords)
            
            prompt_inputs = {
                "question": state.user_query,
                "selected_systems": formatted_guessed_systems,
                "clean_query": state.clean_query,
                "previous_keywords": formatted_keywords,
                "recent_history": formatted_history,
                "chat_summary": state.chat_summary
            }
            
            try:
                # Invoke the keyword expansion chain
                keywords = self.chain.invoke(prompt_inputs) 
                
                # Check if the keywords are present in the clean query
                final_keywords: list[str] = []
                for word in keywords.keywords:
                    if word not in state.clean_query:
                        final_keywords.append(word.lower())
                
                print(f"           ↳ Generated {len(final_keywords)} novel keywords.", file=log_file)
                return {
                    "expanded_keywords": final_keywords
                }
            except Exception as e:
                print(f"\n       └─► [KEYEXPANSION ERROR] Parsing failed: {e}...", file=log_file)
                return {
                    "expanded_keywords": state.expanded_keywords
                }


class HyDENode:
    """
    Generates a hallucinated keyword rich answer for a hallucinated game system
    """
    def __init__(self, chain: RunnableSequence, active_systems: list[str]):
        self.chain = chain
        self.active_systems = active_systems
        
    def __call__(self, state: GameMasterState) -> dict:
        with open(PersistentVars.LOG_FILE, 'a', encoding='utf-8') as log_file:
            print("\n       └─► [NODE: HYDE] Generating a Keyword Dense Hallucination for a random Game System...", file=log_file)   
            
            formatted_history = get_formatted_list(state.recent_history)
            clean_query = set(state.clean_query.split(" "))
            guessed_systems = set(state.guessed_systems)
            active_systems = set(self.active_systems)
            
            prompt_inputs = {
                "question": state.user_query,
                "recent_history":  formatted_history,
                "chat_summary": state.chat_summary
            }
            
            try:
                # Invoke the HyDE chain
                hallucinations = self.chain.invoke(prompt_inputs)
                
                # Check if the keywords are new and add them to clean query
                keys = set(hallucinations.keywords)
                keys |= clean_query
                
                query = " ".join(keys)
                
                # Check and augment the system list with only valid systems
                extracted_systems = set(hallucinations.system)
                extracted_systems = active_systems & extracted_systems
                guessed_systems |= extracted_systems
                
                print(f"           ↳ Generated {len(keys)} novel keywords and {guessed_systems}", file=log_file)
                return {
                    "guessed_systems": list(guessed_systems),
                    "clean_query": query
                }
            except Exception as e:
                print(f"\n       └─► [HYDE ERROR] Parsing failed: {e}...", file=log_file)
                return {
                    "guessed_systems": state.guessed_systems,
                    "clean_query": state.clean_query
                }                      