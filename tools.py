# """

# Script for constructing all of our tools.

# """

from langchain_together.chat_models import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool, BaseToolkit, BaseTool, tool
from typing import List
from langchain_openai import ChatOpenAI


class MathJaxToolkit(BaseToolkit):
    _model: str
    _system_prompt: str

    def __init__(self, model_name: str, inference_server_url: str, **kwargs):
        super().__init__(**kwargs)
        self._model = ChatOpenAI(
            model=model_name,
            base_url=inference_server_url,
            api_key="EMPTY",
            max_tokens=512,
            temperature=0)
        self._system_prompt = """
You are an MathJax expert. Your only job is to format input in MathJax and not to answer questions. You must only answer the instructions with MathJax formatted responses. Please respond to the user's queries in a clear and concise manner and never provide any justification. If you are unable to answer the question, please respond with 'I don't know'. You must make sure that your response is able to compile using MathJax. You must only return the equations in MathJax format and not the solution.
"""

    def _invoke_model(self, prompt: str) -> str:
        return self._model.invoke(
            [SystemMessage(content=self._system_prompt), HumanMessage(content=prompt)]
        )

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="format_equation",
                func=lambda equation: self._invoke_model(
                    f"Format any number of equations that would be helpful for solving this problem: {equation}."
                ),
                description="Formats the given equation into MathJax format.",
            ),
        ]


class PlanningToolkit(BaseToolkit):
    _model: str
    _system_prompt: str

    def __init__(self, model_name: str, inference_server_url: str, **kwargs):
        super().__init__(**kwargs)
        self._model = ChatOpenAI(
            model=model_name,
            base_url=inference_server_url,
            api_key="EMPTY",
            max_tokens=512,
            temperature=0)
        self._system_prompt = """
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
"""

    def _invoke_model(self, prompt: str) -> str:
        return self._model.invoke(
            [SystemMessage(content=self._system_prompt), HumanMessage(content=prompt)]
        )
        
    def get_tools(self) -> List[BaseTool]:
        @tool
        def plan(problem: str) -> str:
            """Generates a plan for guiding the user through the problem solving process for a given math problem description.
            
            Args:
                problem: The math problem that needs to be solved
            """
            return self._invoke_model(
                f"Generate a plan to solve the following problem. Problem: {problem}. Please respond with the plan in a clear and concise manner. You must format the plan in a list"
            )
        
        @tool
        def elaborate(user_input: str) -> str:
            """Generates a plan to address the student's confusion about a problem. 
            
            Args:
                user_input: A description containing both the student's confusion and the problem they're working on
            """
            return self._invoke_model(
                f"Generate a plan to address the student's confusion: {user_input}"
            )
        
        @tool
        def identify_step(context: str) -> str:
            """Identifies what step of the problem the user is on given the plan and their response.
            
            Args:
                context: A description containing the problem, plan, and student's current response
            """
            return self._invoke_model(
                f"Given the following context, identify what step the student is on: {context}"
            )
        
        return [plan, elaborate, identify_step]
    
    # def get_tools(self) -> List[Tool]:
    #     return [
    #         Tool(
    #             name="plan",
    #             func=lambda problem: self._invoke_model(
    #                 f"Generate a plan to solve the following problem. Problem: {problem}. Please respond with the plan in a clear and concise manner. You must format the plan in a list"
    #             ),
    #             description="Generates a plan of guiding the user through the problem solving process for a given math problem description.",
    #         ),
    #         Tool(
    #             name="elaborate",
    #             func=lambda confusion, problem: self._invoke_model(
    #                 f"Generate a plan to address the student's confusion for the problem.\n Confusion: {confusion}\n Problem: {problem}"
    #             ),
    #             description="Generates a plan to address the student's confusion for the problem.",
    #         ),
    #         Tool(
    #             name="step",
    #             func=lambda plan, problem,response: self._invoke_model(
    #                 f"Given the plan and the response, what step is the student on in the problem? \n Problem: {problem}\n Plan: {plan}\n Response: {response}"
    #             ),
    #             description="identifies what step of the problem the user is on given the plan and the response.",
    #         ),
    #         # Tool(
    #         #     name="explain_further",
    #         #     func= lambda step: self._invoke_model(f"There is confusion about this step: {step}. Please explain further."),
    #         #     description="Further explain the given step due to the user's confusion."
    #         # )
    #     ]

