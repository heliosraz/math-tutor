"""

Script for constructing all of our tools.

"""

from langchain_core.tools import tool

# NOTE: when creating tools, you must have a docstring if description not provided
@tool
def multiply(a, b):
    """Multiply two numbers together."""
    return a * b
