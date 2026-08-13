# script that runs a simple agent,
# like the deeplearning ai course, but in a script instead of a notebook, and with a retrieval step


"""agent behaviour:
1 - searches the internet for relevant information about the task
2 - drafts a response to the task, using the retrieved information
3 - critiques the draft, and judges if it needs to be redone
4 - if the draft is not good enough, it goes back to step 1, otherwise it returns the draft
"""

import warnings

# Suppress all deprecation and user warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Specific filters for LangChain deprecations
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*LangChainPendingDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

import logging
import os
import time
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek

# from langchain_core.pydantic_v1 import BaseModel # gives a warning
from pydantic import BaseModel  # new version to avoid warning
from tavily import TavilyClient

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AgentLogger")

# Terminal Color Constants
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
# ===============================================================

# Load variables from .env file into os.environ
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "data")
ATTENTION_PAPER_PATH = os.path.join(DATA_FOLDER_PATH, "1706.03762v7.pdf")
EPUB_PATH = os.path.join(DATA_FOLDER_PATH, "Meditations_of_a_Buddhist_skeptic.epub")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "huggingface")


class AgentState(TypedDict):
    task: str
    draft: str
    critique: str
    retrieval_content: List[str]
    revision_number: int


class Queries(BaseModel):
    queries: List[str]


class Agent:
    def __init__(
        self, max_revisions: int = 3, document_path: str = ATTENTION_PAPER_PATH
    ):
        self.max_revisions = max_revisions
        self.model = ChatDeepSeek(
            model="deepseek-v4-flash",
            temperature=0,
        )
        self.tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.state_graph = self._build_agent_state_graph()
        # self.query_engine = self._init_rag_engine(document_path)

    def _build_agent_state_graph(self) -> StateGraph[AgentState]:
        graph_builder = StateGraph(AgentState)
        # add nodes
        graph_builder.add_node("search", self._search_node)
        graph_builder.add_node("writer", self._writer_node)
        graph_builder.add_node("judge", self._judge_node)
        # add edges
        graph_builder.add_edge("search", "writer")
        graph_builder.add_edge("writer", "judge")
        graph_builder.add_conditional_edges(
            "judge", self._is_answer_text_done, {END: END, "REWRITE": "search"}
        )
        graph_builder.set_entry_point("search")
        # add memory checkpointer
        # memory = SqliteSaver.from_conn_string(":memory:")
        # compile the state graph
        # agent_state_graph = graph_builder.compile(checkpointer=memory)
        agent_state_graph = graph_builder.compile()
        return agent_state_graph

    def _init_rag_engine(self, document_path: str):
        # TODO ... WIP
        logger.debug("DEBUG: Setting up LlamaIndex...")
        # Configure LlamaIndex settings
        Settings.llm = DeepSeek(model="deepseek-v4-flash")
        logger.debug("DEBUG: Setting up HuggingFaceEmbedding model...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", cache_folder=CACHE_DIR
        )
        # Load EPUB document & create RAG index
        logger.debug("DEBUG: Loading EPUB documents...")
        documents = SimpleDirectoryReader(input_files=[EPUB_PATH]).load_data()
        logger.info(f"Loaded {len(documents)} documents.")
        # Create a vector store index from the documents
        index = VectorStoreIndex.from_documents(documents)
        logger.debug(f"created VectorStoreIndex: {index}")
        return index.as_query_engine()

    def _search_node(self, state: AgentState) -> dict:
        logger.info(
            f"{CYAN}--- [SEARCH NODE] (Revision {state['revision_number']}) ---{RESET}"
        )
        SEARCH_PROMPT = (
            "You are a researcher charged with providing information that can "
            "be used whefor writing an answer to the given task below. "
            "Generate a list of search queries that will gather any relevant information. "
            "Generate a JSON object with a 'queries' list containing up to 3 search queries.\n"
            'Example output: {"queries": ["query 1", "query 2"]}'
            "\nzThe task: "
        )

        # use LLM to generate search queries
        start_time = time.time()
        queries = self.model.with_structured_output(method="json_mode").invoke(
            [SystemMessage(content=SEARCH_PROMPT), HumanMessage(content=state["task"])]
            # [HumanMessage(content=state["task"])]
        )
        llm_elapsed = time.time() - start_time
        logger.info(
            f"Query generation took {llm_elapsed:.2f}s | Generated {len(queries.get('queries', []))} queries"
        )

        retrieval_content = state.get("retrieval_content", [])
        start_time = time.time()

        for q in queries["queries"]:
            response = self.tavily_client.search(query=q, max_results=2)
            for r in response["results"]:
                retrieval_content.append(r["content"])
        elapsed = time.time() - start_time
        logger.info(
            f"Retrieved {len(retrieval_content)} total snippets in {elapsed:.2f}s"
        )
        return {
            "retrieval_content": retrieval_content,
        }

    def _writer_node(self, state: AgentState) -> dict:
        logger.info(
            f"{GREEN}--- [WRITER NODE] (Revision {state['revision_number']}) ---{RESET}"
        )
        WRITER_PROMPT = (
            "Write a draft response to the task, using the retrieved information. "
            "If you don't have enough information, write a draft based on your "
            "own knowledge."
        )
        messages = [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=state["task"]),
        ]
        # measure time

        start_time = time.time()
        response = self.model.invoke(messages)
        elapsed = time.time() - start_time
        logger.info(
            f"Draft generated in {elapsed:.2f}s ({len(response.content)} chars)"
        )
        return {
            "draft": response.content,
            "revision_number": state["revision_number"] + 1,
        }

    def _judge_node(self, state: AgentState) -> dict:
        logger.info(
            f"{YELLOW}--- [JUDGE NODE] (Revision {state['revision_number']}) ---{RESET}"
        )
        JUDGE_PROMPT = (
            "You are a text critic, and your main goal is to judge if the draft is good. "
            "Finish your answer with either 'SEARCH_AGAIN', 'REWRITE' or 'ACCEPT'. "
            "DO NOT use the word 'SEARCH_AGAIN', 'REWRITE' or 'ACCEPT' in any other context. "
            "If you think the draft is good enough, write 'ACCEPT'. "
            "If you think the draft needs to be redone, write 'REWRITE'. "
            "If you think the draft needs more information, write 'SEARCH_AGAIN'. "
            "If you choose REWRITE, please provide a short critique of the draft "
            "and what could be improved. "
            "If you choose SEARCH_AGAIN, please say what information is missing and what could be improved."
        )
        user_payload = (
            f"Original Task: {state['task']}\n\nDraft to Evaluate:\n{state['draft']}"
        )
        # TODO make the judge also consider past drafts and critiques
        messages = [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=user_payload),
        ]
        start_time = time.time()
        response = self.model.invoke(messages)
        elapsed = time.time() - start_time
        logger.info(f"Critique finished in {elapsed:.2f}s | Verdict: {response}")
        return {"critique": response.content}

    def _is_answer_text_done(self, state: AgentState):
        if state["revision_number"] > self.max_revisions:
            logger.info(
                f"{MAGENTA}Max revisions ({self.max_revisions}) reached. Terminating graph.{RESET}"
            )
            return END
        elif "REWRITE" in state["critique"]:  # TODO
            logger.info(
                f"{MAGENTA}Judge requested rewrite/search. Routing back to 'search'.{RESET}"
            )
            return "REWRITE"
        else:
            return END

    def run(self, task: str) -> str:
        initial_state: AgentState = {
            "task": task,
            "draft": "",
            "critique": "",
            "retrieval_content": [],
            "revision_number": 0,
        }
        # config = {"configurable": {"thread_id": "1"}}
        # messages = self.state_graph.invoke(initial_state, config=config)
        logger.info("Executing StateGraph...")
        start_time = time.time()
        final_state = self.state_graph.invoke(initial_state)
        total_time = time.time() - start_time
        final_answer = final_state["draft"]
        logger.info(
            f"Graph completed in {total_time:.2f}s across {final_state['revision_number']} iterations."
        )
        breakpoint()
        return final_answer


def main():
    print("========================================")
    print("Welcome to Mr (Elliott) Smith, the Agent")
    print("========================================\n")

    agent = Agent()

    while True:
        task = input("Enter your query/task (or type 'exit' to quit):\n>").strip()
        if not task or task.lower() in ["exit", "quit", "q"]:
            print("ok, ciao!")
            break

        print("\n[Agent starting execution...]")
        answer = agent.run(task)

        print("\n================ FINAL ANSWER ================")
        print(answer)
        print("==============================================\n")


if __name__ == "__main__":
    main()
