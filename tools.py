# """

# Script for constructing all of our tools.

# """

from langchain_core.tools import tool
from langchain_together.chat_models import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseToolkit
from langchain_core.tools import Tool, BaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage


from typing import ClassVar
from langchain_core.tools import Tool, BaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.tools import BaseToolkit, Tool
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = '''
You are an LaTeX expert. You must only ansgwer the instructions with LaTeX formatted responses. Please respond to the user's queries in a clear and concise manner and never provide any justification. If you are unable to answer the question, please respond with 'I don't know'. You must make sure that your response is able to compile using MathJax

###LaTeX Libraries
You may use the following LaTeX libraries in your response:
amsmath
amssymb
tikz
pgfplots
'''

class MathToolkit(BaseToolkit):
    _model: str 
    _system_prompt: str 
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self._model = ChatTogether(
            model=model_name,
            max_tokens=512,
            temperature=0.0,
            verbose=True,
        )
        self._system_prompt = SYSTEM_PROMPT

    def _invoke_model(self, prompt: str) -> str:
        return self.model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ])

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="solve_equation",
                func=lambda problem: self._invoke_model(f"Solve: {problem}"),
                description="Solve a math problem and return the answer."
            ),
            Tool(
                name="explain_solution",
                func=lambda solution: self._invoke_model(f"Explain why this solution makes sense: {solution}"),
                description="Explain the reasoning behind a given math solution."
            )
        ]
        
# model = ChatTogether(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # or whichever chat model you prefer
#     max_tokens=512,                   # required by the completions API :contentReference[oaicite:0]{index=0}
#     temperature=0.0,
#     verbose=True,
# )
# SYSTEM_PROMPT = '''
# You are an LaTeX expert. You must only ansgwer the instructions with LaTeX formatted responses. Please respond to the user's queries in a clear and concise manner and never provide any justification. If you are unable to answer the question, please respond with 'I don't know'. You must make sure that your response is able to compile using MathJax

# ###LaTeX Libraries
# You may use the following LaTeX libraries in your response:
# amsmath
# amssymb
# tikz
# pgfplots
# '''

class LaTeXToolkit(BaseToolkit):
    _model: str 
    _system_prompt: str 
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self._model = ChatTogether(
            model=model_name,
            max_tokens=512,
            temperature=0.0,
            verbose=True,
        )
        self._system_prompt = SYSTEM_PROMPT

    def _invoke_model(self, prompt: str) -> str:
        return self.model.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ])

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="generate_equation",
                func=lambda problem_reasoning: self._invoke_model(f"Generate the equation for this following description: {problem_reasoning}"),
                description="Generates an equation for a given description."
            ),
            Tool(
                name="format_equation",
                func=lambda equation: self._invoke_model(f"Format the equation: {equation}"),
                description="Format the given equation into LaTeX format."
            ),
            Tool(
                name="format_plot",
                func=lambda equation: self._invoke_model(f"Plot this equation: {equation}"),
                description="Plot the given equation in a LaTeX format."
            )
        ]

# @tool
# def generate_equation(problem_reasoning:str):
#     """Generates an equation for a given description"""
#     prompt = f"Generate the equation for this following description: {problem_reasoning}"
#     return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])

# @tool
# def format_equation(equation:str):
#     """Format the given equation into LaTeX format."""
#     prompt = f"Format the equation: {equation}"
#     return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])
    
# @tool
# def format_plot(equation:str):
#     """Plot the given equation in a LaTeX format."""
#     prompt = f"Plot this equation: {equation}"
#     return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])

# @tool
# def plan(problem_reasoning:str):
#     """Plan the given problem reasoning"""
#     prompt = f"Plan the following problem reasoning: {problem_reasoning}"
#     return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])

# @tool
# def solve(problem_reasoning:str):
#     """Solve the given problem reasoning"""
#     prompt = f"Solve the following problem reasoning: {problem_reasoning}"
#     return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])




# class PlanningToolkit(BaseToolkit):
#     @tool
#     def tavily_tool(self):
#         """Creating the tavily tool"""
#         search = TavilySearchResults(max_results=2, description="Use this tool to look up real-time facts like weather, news, or recent events.")
#         return search
    
#     @tool
#     def plan(problem_reasoning:str):
#         """Plan the given problem reasoning"""
#         prompt = f"Plan the following problem reasoning: {problem_reasoning}"
#         return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])

#     @tool
#     def solve(problem_reasoning:str):
#         """Solve the given problem reasoning"""
#         prompt = f"Solve the following problem reasoning: {problem_reasoning}"
#         return model.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=prompt)])


