import getpass
import os
import re

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face_api
if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = tavily_api
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OPENAI_API_KEY"] = openai_api

# Import relevant functionality
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

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
search = TavilySearchResults(max_results=2, description="Use this tool to look up real-time facts like weather, news, or recent events.")
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# other setup we might be missing
model_with_tools = model.bind_tools(tools)

# simpler uses
# prompt = "How's the weather in SF?"
# response1 = model.invoke([SystemMessage(content="Are you able to answer this question without google? Please return yes or no."), HumanMessage(content=prompt)])
# print(response1.content)
# if "no" in response1.content.lower():
#     response = model_with_tools.invoke([HumanMessage(content=prompt)])
#     print(f"ContentString: {response.content}")
#     print(f"ToolCalls: {response.tool_calls}")
# else:
#     print("we do not have to use a tool here")

# simpler uses: use the tool
# response = agent_executor.invoke({
#     "messages": [HumanMessage(content="What's the weather in SF?")],
#     "thread_id": "test-thread-001"  # <- Required for the checkpointer
# })
# response = model_with_tools.invoke([SystemMessage(content="Call a tool to help you answer this question."),
#                                     HumanMessage(content="What's the weather in SF?")])
#
# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

# advanced agent executor
def tool_decider(prompt):
    decision = model.invoke(
        [SystemMessage(content="Are you able to answer this question without google? Please return yes or no."),
         HumanMessage(content=prompt)])

    # if "no" then True, we have to use a tool
    return bool(re.search(r"\bno\b", decision.content.lower()))


# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
prompt1 = "hi im bob! and i live in sf"
if tool_decider(prompt1):
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

prompt2 = "whats the weather where I live?"  # note that memory failed, it didn't know I lived in sf.
if tool_decider(prompt2):
    for step in agent_executor.stream(
        {"messages": [HumanMessage(content=prompt2)]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


