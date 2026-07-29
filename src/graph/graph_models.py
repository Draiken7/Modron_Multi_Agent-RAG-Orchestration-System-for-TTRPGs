from langchain_ollama import ChatOllama


from src.env import PersistentVars

class ModelManager:
    """
    Model manager to customize models used for this system. Simply saves and returns a langchain supported model object
    """
    def __init__ (self):
        self.router = None# Model for the Query cleaning and system guessing
        self.rcritic = None# Critic model for router
        self.summarizer = None# Model to run the summarization chain in case of phase 1 routing failure
        self.gm = None # Model that acts as the GM to answer the query with retieved context
        self.gmcritic = None # Critic model for gm
        self.memory = None # Model that summarizes the recent chat for long term memory
        self.systemca = None # Re-guesses the Systems for better retrieval. IDEALLY SHOULD BE A BETTER MODEL THAN THE ROUTER.
        self.keyexpansion = None # Geenrates a list of keyword (jargon) for better retrieval
        self.hyde = None # A hallucination generation model for keyword (jargon) and system infusion. IDEALLY SHOULD BE AN EXPENSIVE MODEL USED AS A FINAL TRY BEFORE GRACEFUL FAILURE.

        
        # Initializations
        with open(PersistentVars.LOG_FILE, "a", encoding='utf-8') as log_file:
            print("[SYSTEM] Summoning Models...", file=log_file)
            try:
                self.router = ChatOllama(model='qwen3:8b ', temperature=0.0)
                # self.rcritic = ChatOllama(model='qwen3.5:4b', temperature=0.0, keep_alive='0')
                # self.summarizer = ChatOllama(model='llama3.1:8b', temperature=0.3, keep_alive='0')
                self.rcritic = self.router
                # self.rcritic = ChatGoogleGenerativeAI(model="gemini-3-flash-live", temperature=0.0)
                self.summarizer = self.router
                self.gm = self.router
                self.gmcritic = self.router
                self.memory = self.router
                self.systemca = self.router
                self.keyexpansion = self.systemca
                self.hyde = self.router
                print("  [SYSTEM] Models Ready.", file=log_file)
            except Exception as e:
                print(f"  --[SYSTEM ERROR] Model Initialization failed: {e}", file=log_file)
        