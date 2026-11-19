"""Two tools a model can ask for: a course lookup and a safe calculator. All data is invented."""

import ast
import operator
from collections.abc import Callable
from typing import Any

COURSES: dict[str, dict[str, Any]] = {
    "CS 101": {"title": "Intro to Programming", "students": 84, "sections": 3},
    "CS 230": {"title": "Data Structures", "students": 45, "sections": 2},
    "CS 350": {"title": "Operating Systems", "students": 36, "sections": 1},
}


def _squash(code: str) -> str:
    """'cs 101', 'CS101' and ' Cs  101 ' all become 'CS101'."""
    return "".join(code.split()).upper()


def get_course_info(course: str) -> dict[str, Any]:
    wanted = _squash(course)
    for code, info in COURSES.items():
        if _squash(code) == wanted:
            return {"course": code, **info}
    raise LookupError(f"unknown course {course!r}; known courses: {', '.join(COURSES)}")


_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    """Evaluate one tree node. Only numbers and + - * / are allowed; all else is refused."""
    # type(...) in (int, float) rather than isinstance, because True is an int in Python.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    kind = type(node.op if isinstance(node, ast.BinOp | ast.UnaryOp) else node).__name__
    raise ValueError(f"not allowed in a calculation: {kind}")


def calculate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression: {expression!r}") from exc
    try:
        return float(_eval(tree.body))
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc


TOOLS: dict[str, Callable[..., Any]] = {
    "get_course_info": get_course_info,
    "calculate": calculate,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_course_info",
            "description": (
                "Look up a course by its code, for example 'CS 101'. Returns the title, "
                "the number of students and the number of sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course": {"type": "string", "description": "Course code, like 'CS 101'."}
                },
                "required": ["course"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate an arithmetic expression. Use this for every calculation, even simple "
                "ones. Supports + - * / and parentheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "For example '84 / 3'."}
                },
                "required": ["expression"],
            },
        },
    },
]
