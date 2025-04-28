import re
from utils import load_credentials

load_credentials()

from agent import Agent

gen_agent = Agent(model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")



