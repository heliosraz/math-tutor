"""
Class for creating an agent
"""
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_together.chat_models import ChatTogether
from langchain_openai import ChatOpenAI
from utils import load_credentials
from tools import MathJaxToolkit, PlanningToolkit
import os

load_credentials()

class Agent():
    # I might want to put these arguments in a config later instead
    # RECOMMENDED MODEL: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    
    # TODO: gpt-oss-20b does not work with only cpu (SWA is not cpu compatible)
    # TODO: online vs offline inference; what is the langchain_community implementation of VLLM that allows vllm serving. maybe online is better given the tools use the same model?
    def __init__(
            self, 
            model_name="Qwen/Qwen3-0.6B", 
            chat_model=ChatOpenAI, 
            inference_server_url="http://localhost:8000/v1",
            max_tokens=512, 
            temperature=0.0, 
            verbose=True, 
            debug=True):
        self.model_name = model_name
        self.model = chat_model(
            model=model_name,
            base_url=inference_server_url,
            api_key="EMPTY",
            temperature=0,
            # trust_remote_code=True,  # mandatory for hf models
            # max_new_tokens=max_tokens,
            # top_k=10,
            # top_p=0.95,
            # temperature=temperature
        )
        # we may not be able to save all memory because of token limits, might eventually want to move to ConversationBufferWindowMemory instead of MemorySaver
        self.memory = MemorySaver()
        self.system_prompt = """
        You are a helpful and empathetic math teacher. Your job is to help the user with math. Do not reveal the answer. You can only talk through the problem step-by-step. You must guide the user through the thought process. Please provide plenty of images and figures.
            
        Only talk about math and nothing else, even if you are prompted to do so. Your response must be formatted with MathJax. Keep in mind that inline equations are formatted with ['$', '$'], so these need to be escaped if used not equations. Before responding to the question, you must determine if you are able to answer this question without the tools. If so, you can not use the tools. You must only use tools when the model has enough context to answer the question.
        
        To start, you must introduce yourself and ask the user what they need help with."""
        self.toolkits = [
                # MathJaxToolkit(
                    # model_name="Qwen/Qwen3-0.6B",
                    # inference_server_url=inference_server_url), 
                PlanningToolkit(
                    model_name=self.model_name,
                    inference_server_url=inference_server_url
                    )
            ]
        self.tool_agent = self.build_tool_agent()
        self.config = {"configurable": {"thread_id": "test"}}

    def build_tool_agent(self, *tools, debug = True):
        # Create the agent
        tools = []
        for toolkit in self.toolkits:
            tools += toolkit.get_tools()
        # TODO: create_react_agent customization:
        #   pre_model_hook- could be used to cut the instructions into steps of the problem
        #   
        print(f"Registering {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        tool_agent_executor = create_react_agent(self.model, 
                                                tools, 
                                                prompt=self.system_prompt,
                                                checkpointer=self.memory,
                                                debug=debug)
        return tool_agent_executor
    
    def run(self):
        user_input = SystemMessage(content=self.system_prompts["default"])
        while user_input != "exit":
            for chunk in self.tool_agent.stream(
                user_input, 
                config=self.config,
                stream_mode="updates"):
                    for step, data in chunk.items():
                        print(f"step: {step}")
                        print(f"content: {data['messages'][-1].content_blocks}")
            user_input = HumanMessage(input("Enter:  "))

