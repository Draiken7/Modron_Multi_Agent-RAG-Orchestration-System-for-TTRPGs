from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from src.env import GraphAgentPrompts, PersistentVars
from src.graph.graph_models import ModelManager
from src.graph.graph_database import DatabaseOP
import src.graph.graph_nodes as nodes
from src import schema



class PromptRegistry:
    """
    class to handle prompt registery
    """
    def __init__(self):
        with open(PersistentVars.LOG_FILE, "a", encoding="utf-8") as log_file:
            print("    [SYSTEM] Loading prompt Templates...", file=log_file)
            
            try:
                self.router_prompt = ChatPromptTemplate.from_messages([
                ("system", GraphAgentPrompts.ROUTER_SYSTEM_PROMPT),
                ("human", GraphAgentPrompts.ROUTER_USER_PROMPT)
                ])

                self.grader_prompt = ChatPromptTemplate.from_messages([
                        ("system", GraphAgentPrompts.ROUTER_GRADER_SYSTEM),
                        ("human", GraphAgentPrompts.ROUTER_GRADER_USER)
                ])

                self.summarizer_prompt = ChatPromptTemplate.from_messages([
                    ('system', GraphAgentPrompts.SUMMARIZER_SYSTEM_PROMPT),
                    ('human', GraphAgentPrompts.SUMMARIZER_USER_PROMPT)
                ])

                self.gm_prompt = ChatPromptTemplate.from_messages([
                    ("system", GraphAgentPrompts.GAME_MASTER_SYSTEM_PROMPT),
                    ("human", GraphAgentPrompts.GAME_MASTER_USER_PROMPT)
                ])

                self.memory_prompt = ChatPromptTemplate.from_messages([
                    ("human", GraphAgentPrompts.MEMORY_SUMMARY_PROMPT)
                ])

                self.gmcritic_prompt = ChatPromptTemplate.from_messages([
                    ("system", GraphAgentPrompts.GM_GRADER_SYSTEM),
                    ("human", GraphAgentPrompts.GM_GRADER_USER)
                ])

                self.systemca_prompt = ChatPromptTemplate.from_messages([
                    ("system", GraphAgentPrompts.SYSTEMCA_SYSTEM_PROMPT),
                    ("human", GraphAgentPrompts.SYSTEMCA_USER_PROMPT)
                ])

                self.keyexpansion_prompt = ChatPromptTemplate.from_messages([
                    ("system", GraphAgentPrompts.KEYEXPANSION_SYSTEM_PROMPT),
                    ("human", GraphAgentPrompts.KEYEXPANSION_USER_PROMPT)
                ])

                self.hyde_prompt = ChatPromptTemplate.from_messages([
                    ("system", GraphAgentPrompts.HYDE_SYSTEM_PROMPT),
                    ("human", GraphAgentPrompts.HYDE_USER_PROMPT)
                ])
            except Exception as e:
                print(f"        ----[ERROR] Prompt Registery error: {e}", file=log_file)
 
        
class ChainRegistry:
    """
    Class to handle all LCEL chain regisry.
    """
    def __init__(self, modelDispenser: ModelManager, promptRegistry: PromptRegistry):
        with open(PersistentVars.LOG_FILE, "a", encoding="utf-8") as log_file:
            print("    [SYSTEM] Compiling LCEL chains...", file=log_file)
            try:    
                self.router_chain = promptRegistry.router_prompt | modelDispenser.router.with_structured_output(schema.QueryIntent)
                self.grader_chain = promptRegistry.grader_prompt | modelDispenser.rcritic.with_structured_output(schema.RouterGraderOutput)
                self.summary_chain = promptRegistry.summarizer_prompt | modelDispenser.summarizer | StrOutputParser()
                self.gm_chain = promptRegistry.gm_prompt | modelDispenser.gm | StrOutputParser()
                self.memory_chain = promptRegistry.memory_prompt | modelDispenser.memory | StrOutputParser()
                self.gmcritic_chain = promptRegistry.gmcritic_prompt | modelDispenser.gmcritic.with_structured_output(schema.GMGraderOutput)
                self.systemca_chain = promptRegistry.systemca_prompt | modelDispenser.router.with_structured_output(schema.SystemCAOutput)
                self.keyexpansion_chain = promptRegistry.keyexpansion_prompt | modelDispenser.keyexpansion.with_structured_output(schema.KeyExpansionOutput)
                self.hyde_chain = promptRegistry.hyde_prompt | modelDispenser.hyde.with_structured_output(schema.HyDEOutput)
            except Exception as e:
                print(f"        ----[ERROR] Chain Registery error: {e}", file=log_file)
                   
       
class WorkerRegistry:
    """
    Class to handle worker registery
    """
    def __init__(self, chain: ChainRegistry, db: DatabaseOP):
        with open(PersistentVars.LOG_FILE, "a", encoding="utf-8") as log_file:
            print("    [SYSTEM] Assembling workers...", file=log_file)
            try:
                self.router_worker = nodes.RouterNode(chain=chain.router_chain, active_systems=schema.ACTIVE_SYSTEMS)
                self.critic_worker = nodes.RouterCriticNode(chain=chain.grader_chain, active_systems=schema.ACTIVE_SYSTEMS)
                self.routerfail_worker = nodes.RouterFailSafeNode()
                self.retriever_worker = nodes.RetrieverNode(active_systems=schema.ACTIVE_SYSTEMS, db=db)
                self.summarizer_worker = nodes.SummarizerNode(chain=chain.summary_chain, active_systems=schema.ACTIVE_SYSTEMS)
                self.memory_worker = nodes.MemoryNode(chain=chain.memory_chain)
                self.gm_worker = nodes.GameMasterNode(chain=chain.gm_chain)
                self.gmcritic_worker = nodes.GMCriticNode(chain=chain.gmcritic_chain)
                self.gmfail_worker = nodes.GMFailSafeNode()
                self.systemca_worker = nodes.SystemCANode(chain=chain.systemca_chain, active_systems=schema.ACTIVE_SYSTEMS)
                self.keyexpansion_worker = nodes.KeyExpansionNode(chain=chain.keyexpansion_chain)
                self.hyde_worker = nodes.HyDENode(chain=chain.hyde_chain, active_systems=schema.ACTIVE_SYSTEMS)
            except Exception as e:
                print(f"        ----[ERROR] Worker Registery error: {e}", file=log_file)
 
       
class PipelineRegistry:
    """
    Handles the pipeline registry from a single entry point. Saves instantiated registry objects.
    """
    def __init__(self, modelDispenser: ModelManager, db: DatabaseOP):
        with open(PersistentVars.LOG_FILE, "a", encoding="utf-8") as log_file:
            print("[SYSTEM] Collating Pipeline...", file=log_file)
            log_file.flush()
            try:
                self.prompts = PromptRegistry()
                self.chains = ChainRegistry(modelDispenser=modelDispenser, promptRegistry=self.prompts) 
                self.workers = WorkerRegistry(chain=self.chains, db=db)
                print("[SYSTEM] Pipeline Ready.", file=log_file)
            except Exception as e:
                print(f"  --[SYSTEM ERROR] Pipeline initialization Failed: {e}", file=log_file)