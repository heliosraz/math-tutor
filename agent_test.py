import re
from utils import load_credentials

load_credentials()

# Import relevant functionality
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from tools import generate_equation, format_equation, format_plot
from langchain.agents import create_tool_calling_agent

# Create the agent
memory = MemorySaver()

# from langchain.chat_models import init_chat_model
# model = init_chat_model("gpt-4", model_provider="openai")
from langchain_together.chat_models import ChatTogether

model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # or whichever chat model you prefer
    max_tokens=512,                   # required by the completions API :contentReference[oaicite:0]{index=0}
    temperature=0.0,
    verbose=True,
)
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )
# model = ChatHuggingFace(llm=llm)

# model = ChatHuggingFace(llm=llm)
# search = TavilySearchResults(max_results=2, description="Use this tool to look up real-time facts like weather, news, or recent events.")
# tools = [search]
tools = [generate_equation, format_equation, format_plot]

agent_executor = create_react_agent(model, tools, checkpointer=memory)
# other setup we might be missing
model_with_tools = model.bind_tools(tools)

# advanced agent executor
def tool_decider(prompt):
    decision = agent_executor.invoke(
        [SystemMessage(content="Are you able to answer this question without a web search? Please return yes or no. You must only use tools when the model has enough context to answer the question. If you are unsure, say you are unsure."),
         HumanMessage(content=prompt)])

    # if "no" then True, we have to use a tool
    return bool(re.search(r"\bno\b", decision.content.lower()))


# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
messages = [SystemMessage(content="You are a helpful math assistant. Your response must be formatted with LaTeX. Before responding to the question, you must determine if you are able to answer this question without LaTeX. You must only use tools when the model has enough context to answer the question. If you are unsure, say you are unsure."), HumanMessage(content="hi! I need math help") ]

# If not, do not use the tools.
user_input = ""
while user_input != "exit":
    for step in agent_executor.stream(
            {"messages": messages},
            config,
            stream_mode="values",
        ):
        step["messages"][-1].pretty_print()
    user_input = input("Enter:  ")
    messages.append(HumanMessage(content=user_input))

# prompt1 = "hi im bob! and i live in sf"
# if tool_decider(prompt1):
#     for step in agent_executor.stream(
#         {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
#         config,
#         stream_mode="values",
#     ):
#         step["messages"][-1].pretty_print()

# prompt2 = "whats the weather where I live?"  # note that memory failed, it didn't know I lived in sf.
# if tool_decider(prompt2):
#     for step in agent_executor.stream(
#         {"messages": [HumanMessage(content=prompt2)]},
#         config,
#         stream_mode="values",
#     ):
#         step["messages"][-1].pretty_print()


