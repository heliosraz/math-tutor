from flask import Flask, render_template, request, url_for, jsonify
from langchain_core.messages import HumanMessage, SystemMessage
from agent import Agent
from utils import load_credentials
import time

app = Flask(__name__)
agent = Agent("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
config = {"configurable": {"thread_id": "abc123"}}
messages = [SystemMessage(content=agent.system_prompts["default"])]
# create text file to store conversation
convo_txt = "./conversations/" + time.ctime().replace(' ', '_').replace(':', '_')


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
    with open(convo_txt, 'a') as f:
        f.write('User: ' + user_msg + '\n')
        f.write('Bot: ' + response + '\n')
    return jsonify({'response':response})




if __name__ == "__main__":
    load_credentials()
    app.run(host='0.0.0.0', port=5000, debug=True)
