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


# class MathToolkit(BaseToolkit):
#     _model: str 
#     _system_prompt: str 
#     def __init__(self, model_name: str, **kwargs):
#         super().__init__(**kwargs)
#         self._model = ChatTogether(
#             model=model_name,
#             max_tokens=512,
#             temperature=0.0,
#             verbose=True,
#         )
#         self._system_prompt = SYSTEM_PROMPT

#     def _invoke_model(self, prompt: str) -> str:
#         return self._model.invoke([
#             SystemMessage(content=self.system_prompt),
#             HumanMessage(content=prompt)
#         ])

#     def get_tools(self) -> List[Tool]:
#         return [
#             Tool(
#                 name="solve_equation",
#                 func=lambda problem: self._invoke_model(f"Solve: {problem}"),
#                 description="Solve a math problem and return the answer."
#             ),
#             Tool(
#                 name="explain_solution",
#                 func=lambda solution: self._invoke_model(f"Explain why this solution makes sense: {solution}"),
#                 description="Explain the reasoning behind a given math solution."
#             )
#         ]
        
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

class MathJaxToolkit(BaseToolkit):
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
        self._system_prompt = '''
You are an LaTeX expert. Your only job is to format input in LaTeX and not to answer questions. You must only answer the instructions with LaTeX formatted responses. Please respond to the user's queries in a clear and concise manner and never provide any justification. If you are unable to answer the question, please respond with 'I don't know'. You must make sure that your response is able to compile using MathJax. You must only return the equations in LaTeX format and not the solution.

###LaTeX Libraries
You may use the following LaTeX libraries in your response:
amsmath
amssymb
tikz
pgfplots
'''

    def _invoke_model(self, prompt: str) -> str:
        return self._model.invoke([
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ])

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="format_equation",
                func=lambda equation: self._invoke_model(f"Format any number of equations that would be helpful for solving this problem: {equation}."),
                description="Extracts and formats equation the given equation into LaTeX format."
            ),
            Tool(
                name="format_plot",
                func=lambda equation: self._invoke_model(f"Plot this equation: {equation}"),
                description="Plot the given equation in a LaTeX format."
            )
        ]




class PlanningToolkit(BaseToolkit):
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
        self._system_prompt = '''
You are an curriculum expert. Your job is to make a plan of answering and solving the student's question, without fully answering the problem. Please consider what. Please respond to the user's queries in a clear and concise manner and provide justification if needed. If you are unable to answer the question, please respond with 'I don't know'. 

###Consider:
- The student's question
- The student's level of understanding
- The student's learning goals
- The student's confusion

###Topics
These are the following topics students may ask about and are expected to know:
- Algebra
- Geometry
- Trigonometry
- Probability
- Statistics
'''

    def _invoke_model(self, prompt: str) -> str:
        return self._model.invoke([
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ])

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="plan",
                func=lambda problem: self._invoke_model(f"Generate a plan to solve the following problem: {problem}"),
                description="Generates a plan to the solution for a given problem description."
            ),
            Tool(
                name="elaborate",
                func=lambda confusion, problem: self._invoke_model(f"Generate a plan to address the student's confusion for the problem.\n Confusion: {confusion}\n Problem: {problem}"),
                description="Generates a plan to address the student's confusion for the problem."
            ),
            Tool(
                name="step",
                func=lambda answer: self._invoke_model(f"What step is the student on in the following: {answer}"),
                description="identifies what step the user is on in the problem solving process."
            ),
            # Tool(
            #     name="explain_further",
            #     func= lambda step: self._invoke_model(f"There is confusion about this step: {step}. Please explain further."),
            #     description="Further explain the given step due to the user's confusion."
            # )
        ]
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


