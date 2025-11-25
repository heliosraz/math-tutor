"""

Class for testing a series of prompts with a conversation on the agent

"""

from agent import Agent
import time

config = {"configurable": {"thread_id": "abc123"}}
agent = Agent()

def invoke(prompt):
    for step in agent.stream(
            prompt,
            stream_mode="values",
            config=config
    ):
        step["messages"][-1].pretty_print()

prompt0 = "what is 3 multiplied by 7? Please use the planning tool"
invoke(prompt0)
print("sleeping for a little, I can only process so many messages at once!")
time.sleep(10)
prompt1 = "hi im bob! and i live in sf"
invoke(prompt1)
print("sleeping for a little, I can only process so many messages at once!")
time.sleep(10)
prompt2 = "whats the weather where I live?"
invoke(prompt2)


