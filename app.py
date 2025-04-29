from flask import Flask, render_template, request, url_for, jsonify
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from agent import Agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from utils import load_credentials
from langchain_together.chat_models import ChatTogether


# toolkit = [generate_equation, format_equation, format_plot, explain_further]
# memory = MemorySaver()
# model = ChatTogether(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#     max_tokens=512,
#     temperature=0.0,
#     verbose=True,
# )
# agent_executor = create_react_agent(model=model, tools=toolkit, checkpointer=memory)
# agent = Agent("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
# messages = [SystemMessage(content="You are a helpful math assistant. Your response must be formatted with LaTeX. Before responding to the question, you must determine if you are able to answer this question without LaTeX. You must only use tools when the model has enough context to answer the question. If you are unsure, say you are unsure.")]
app = Flask(__name__)
agent = Agent("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
config = {"configurable": {"thread_id": "abc123"}}
messages = [SystemMessage(content=agent.system_prompts["default"])]


@app.route("/")
def root():
    return render_template('index.html')


@app.route("/chat", methods=["POST"])
def get_tutor_response():
    user_msg = request.json['message']
    # this is a placeholder because i'm not really sure it works this way
    # plus casting generator to list every time is bad
    messages.append(HumanMessage(content=user_msg))
    steps = list(agent.tool_agent.stream(
            {"messages": messages},
            config,
            stream_mode="values"))
    response = steps[-1]["messages"][-1].pretty_repr()
    return jsonify({'response':response})


if __name__ == "__main__":
    load_credentials()
    app.run(debug=True)
