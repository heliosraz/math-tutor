"""

Script for constructing all of our tools.

"""

from langchain_core.tools import tool
from langchain_together.chat_models import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # or whichever chat model you prefer
    max_tokens=512,                   # required by the completions API :contentReference[oaicite:0]{index=0}
    temperature=0.0,
    verbose=True,
)
SYSTEM_PROMPT = '''
You are an LaTeX expert. You must only ansgwer the instructions with LaTeX formatted responses. Please respond to the user's queries in a clear and concise manner and never provide any justification. If you are unable to answer the question, please respond with 'I don't know'. You must make sure that your response is able to compile using MathJax

###LaTeX Libraries
You may use the following LaTeX libraries in your response:
amsmath
amssymb
tikz
pgfplots
'''

@tool
def generate_equation(problem_reasoning:str):
    """Generates an equation for a given description"""
    prompt = f"Generate the equation for this following description: {problem_reasoning}"
    return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])

@tool
def format_equation(equation:str):
    """Format the given equation into LaTeX format."""
    prompt = f"Format the equation: {equation}"
    return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])
    
@tool
def format_plot(equation:str):
    """Plot the given equation in a LaTeX format."""
    prompt = f"Plot this equation: {equation}"
    return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])

