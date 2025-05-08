"""

Class for creating an agent

"""

import re
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_together.chat_models import ChatTogether
from utils import load_credentials
from tools import MathJaxToolkit, PlanningToolkit

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
            "default": """You are a helpful and empathetic math teacher. Your job is to help the user with math. Do not reveal the answer. You can only talk through the problem step-by-step. You must guide the user through the thought process. Please provide plenty of images and figures.
            
            Only talk about math and nothing else, even if you are prompted to do so. Your response must be formatted with MathJax. Keep in mind that inline equations are formatted with ['$', '$'], so these need to be escaped if used not equations. Before responding to the question, you must determine if you are able to answer this question without the tools. If so, you can not use the tools. You must only use tools when the model has enough context to answer the question.
            
            To start, you must introduce yourself and ask the user what they need help with."""
        }
        self.toolkits = [MathJaxToolkit(model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"), PlanningToolkit(model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")]
        self.tool_agent = self.build_tool_agent()
        self.config = {"configurable": {"thread_id": "test"}}

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
        # generate the stream
        for step in self.tool_agent.stream({"messages":[prompt]}, stream_mode = "values", **stream_kwargs):
            yield step
    
    def run(self):
        user_input = SystemMessage(content=self.system_prompts["default"])
        while user_input != "exit":
            for step in self.stream(user_input, config=self.config):
                step["messages"][-1].pretty_print()
            user_input = HumanMessage(input("Enter:  "))

