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
from utils import load_credentials
from tools import MathToolkit

load_credentials()

class Agent():
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
            "default": "You are a helpful math assistant. Your response must be formatted with LaTeX. Before responding to the question, you must determine if you are able to answer this question without LaTeX. You must only use tools when the model has enough context to answer the question. If you are unsure, say you are unsure.",
            "tavily": "Are you able to answer this question without google? Please return yes or no.",
            "multiply": "Are you able to answer this question without calculator? Please return yes or no."
        }
        self.toolkits = [MathToolkit(model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")]
        tool_agent = self.build_tool_agent()

    def build_tool_agent(self, *tools):
        # Create the agent
        tools = []
        for toolkit in self.toolkits:
            tools += toolkit.get_tools()
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

        # # decide which tools to use
        # if self._tool_decider(prompt, self.system_prompts["tavily"]):
        #     executor = self.build_tool_agent(self.tavily_tool())
        # elif self._tool_decider(prompt, self.system_prompts["multiply"]):
        #     executor = self.build_tool_agent(multiply)
        # else:
        #     executor = self.build_tool_agent()
        system_prompt = SystemMessage(content=self.system_prompts["default"])
        prompt = HumanMessage(content=self.system_prompts["default"])
        executor = self.build_tool_agent()

        # generate the stream
        for step in executor.stream({"messages":[system_prompt, prompt]}, **stream_kwargs):
            yield step
    
    def run(self):
        config = {}
        messages = [
            SystemMessage(content=self.system_prompts["default"])
        ]
        user_input = ""
        while user_input != "exit":
            for step in self.agent_executor.stream(
                    {"messages": messages},
                    config,
                    stream_mode="values",
                ):
                step["messages"][-1].pretty_print()
            user_input = input("Enter:  ")
            messages.append(HumanMessage(content=user_input))
