# script that runs a simple agent,
# like the deeplearning ai course, but in a script instead of a notebook, and with a retrieval step


"""agent behaviour:
1 - searches the internet for relevant information about the task
2 - drafts a response to the task, using the retrieved information
3 - critiques the draft, and judges if it needs to be redone
4 - if the draft is not good enough, it goes back to step 1, otherwise it returns the draft
"""

import time
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient

from langchain_deepseek import ChatDeepSeek
from typing import TypedDict, List
from langchain_core.pydantic_v1 import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver

# Load variables from .env file into os.environ
load_dotenv()


class AgentState(TypedDict):
    task: str
    draft: str
    critique: str
    retrieval_content: List[str]
    revision_number: int


class Queries(BaseModel):
    queries: List[str]


class Agent:
    def __init__(self, max_revisions: int = 3):
        self.max_revisions = max_revisions
        self.model = ChatDeepSeek(
            model="deepseek-v4-flash",
            temperature=0,
        )
        self.s = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.state_graph = self._build_agent_state_graph()

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
        memory = SqliteSaver.from_conn_string(":memory:")
        # compile the state graph
        # agent_state_graph = graph_builder.compile(checkpointer=memory)
        agent_state_graph = graph_builder.compile()
        return agent_state_graph

    def _search_node(self, state: AgentState) -> dict:
        print("DEBUG search node, revision_number:", state["revision_number"])
        return {
            "retrieval_content": "TODO retrieval content ... revision {state['revision_number']}"
        }
        # TODO
        # TODO create a better search query based on the task, the draft, and the critique
        queries = self.model.with_structured_output(Queries).invoke(
            [SystemMessage(content=SEARCH_PROMPT), HumanMessage(content=state["task"])]
        )
        content = state["content"] or []
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])

    def _writer_node(self, state: AgentState) -> dict:
        print("DEBUG writer node, revision_number:", state["revision_number"])
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
        end_time = time.time()
        print(f"DEBUG deepseekl call time elapsed: {end_time - start_time:.2f} seconds")
        return {
            "draft": response.content,
            "revision_number": state["revision_number"] + 1,
        }

    def _judge_node(self, state: AgentState) -> dict:
        print("DEBUG judge node, revision_number:", state["revision_number"])
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
        response = self.model.invoke(messages)
        return {"critique": response.content}

    def _is_answer_text_done(self, state: AgentState):
        if state["revision_number"] > self.max_revisions:
            return END
        elif "REWRITE" in state["critique"]:  # TODO
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
        messages = self.state_graph.invoke(initial_state)
        print(f"After {messages['revision_number']} revisions...")
        breakpoint()
        final_answer = messages["draft"]
        return final_answer


def main():
    print("========================================")
    print("Welcome to the Smith, the Agent")
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
