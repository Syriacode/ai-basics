import pytest

from ai_basics.tools import TOOL_SCHEMAS, TOOLS, calculate, get_course_info


def test_get_course_info_returns_the_paper_numbers():
    assert get_course_info("CS 101") == {
        "course": "CS 101",
        "title": "Intro to Programming",
        "students": 84,
        "sections": 3,
    }


@pytest.mark.parametrize("spelling", ["cs101", "CS101", "cs 101", " Cs  101 "])
def test_get_course_info_ignores_case_and_spaces(spelling):
    assert get_course_info(spelling)["course"] == "CS 101"


def test_get_course_info_unknown_course_names_the_known_ones():
    with pytest.raises(LookupError) as error:
        get_course_info("CS 999")
    message = str(error.value)
    assert "CS 101" in message
    assert "CS 230" in message
    assert "CS 350" in message


def test_calculate_students_per_section():
    assert calculate("84 / 3") == 28


def test_calculate_follows_parentheses():
    assert calculate("(84 + 45) / 5") == pytest.approx(25.8)


def test_calculate_precedence_and_negative_numbers():
    assert calculate("2 + 3 * 4") == 14
    assert calculate("-3 + 5") == 2


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "import('os').system('echo hi')",
        "2 ** 100",
        "abs(-3)",
        "x + 1",
        "True + 1",
        "'a' * 3",
        "1 / 0",
        "84 /",
        "",
    ],
)
def test_calculate_refuses_everything_else(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_schema_names_are_exactly_the_tool_names():
    schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert schema_names == set(TOOLS)


def test_schemas_are_in_ollama_format():
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        parameters = schema["function"]["parameters"]
        assert parameters["type"] == "object"
        assert set(parameters["required"]) <= set(parameters["properties"])


def test_calculate_description_says_to_use_it_for_every_calculation():
    by_name = {schema["function"]["name"]: schema["function"] for schema in TOOL_SCHEMAS}
    assert "every calculation" in by_name["calculate"]["description"]
