"""

Class for creating an agent

"""

import os
import re
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_together.chat_models import ChatTogether

from tools import multiply

# environment setups with APIs
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face_api
if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = tavily_api
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OPENAI_API_KEY"] = openai_api

class Agent:
    # I might want to put these arguments in a config later instead
    # RECOMMENDED MODEL: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    def __init__(self, model_name, chat_model=ChatTogether, max_tokens=512, temperature=0.0, verbose=True):
        self.model_name = model_name
        self.model = chat_model(
            model_name=model_name,
            max_tokens=max_tokens, # required by the completions API :contentReference[oaicite:0]{index=0}
            temperature=temperature,
            verbose=verbose
        )
        # we may not be able to save all memory because of token limits, might eventually want to move to ConversationBufferWindowMemory instead of MemorySaver
        self.memory = MemorySaver()
        # the following are for deciding which tools we should use
        self.system_prompts = {
            "tavily": "Are you able to answer this question without google? Please return yes or no.",
            "multiply": "Are you able to answer this question without calculator? Please return yes or no.",
        }

    # todo: consider moving this to tools.py somehow
    def tavily_tool(self):
        """Creating the tavily tool"""
        search = TavilySearchResults(max_results=2, description="Use this tool to look up real-time facts like weather, news, or recent events.")
        return search

    def build_tool_agent(self, *tools):
        # Create the agent
        tools = list(tools)
        tool_agent_executor = create_react_agent(self.model, tools, checkpointer=self.memory)

        return tool_agent_executor

    def _tool_decider(self, prompt, system_prompt):
        """Decides if we should use the tool by having the model reason whether it should use its tools while it doesn't have it's agent and memory attached"""
        decision = self.model.invoke(
            [SystemMessage(content=system_prompt),
             HumanMessage(content=prompt)])

        # if "no" then True, we have to use a tool
        # response = step["messages"][-1]
        return bool(re.search(r"\bno\b", decision.content.lower()))

    def stream(self, prompt, **stream_kwargs):
        """Normal stream but reasons which tools it should use first"""

        # decide which tools to use
        if self._tool_decider(prompt, self.system_prompts["tavily"]):
            executor = self.build_tool_agent(self.tavily_tool())
        elif self._tool_decider(prompt, self.system_prompts["multiply"]):
            executor = self.build_tool_agent(multiply)
        else:
            executor = self.build_tool_agent()

        # generate the stream
        for step in executor.stream({"messages":[prompt]}, **stream_kwargs):
            yield step
