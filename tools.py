"""

Script for constructing all of our tools.

"""

from langchain_core.tools import tool
from langchain_together.chat_models import ChatTogether

model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # or whichever chat model you prefer
    max_tokens=512,                   # required by the completions API :contentReference[oaicite:0]{index=0}
    temperature=0.0,
    verbose=True,
)
SYSTEM_PROMPT = '''
You are an LaTeX expert. You must only answer the instructions with LaTeX formatted responses. Please respond to the user's queries in a clear and concise manner and never provide any justification. If you are unable to answer the question, please respond with 'I don't know'. You must make sure that your response is able to compile using MathJax

###LaTeX Libraries
You may use the following LaTeX libraries in your response:
amsmath
amssymb
tikz
'''

@tool
def gather_equation(problem_reasoning:str):
    """Generates an equation for a given description"""
    prompt = f"Generate the equation for this following description: {problem_reasoning}"
    return model.invoke(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=512,
    )

@tool
def format_equation(equation:str):
    """Format the given equation into LaTeX format."""
    prompt = f"Format the equation: {equation}"
    return model.invoke(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=512,
    )
    
@tool
def format_plot(equation:str):
    """Format the given equation into LaTeX format."""
    prompt = f"Plot this equation: {equation}"
    return model.invoke(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=512,
    )
