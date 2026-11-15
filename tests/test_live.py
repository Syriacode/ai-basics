"""Tests that call the real model (Ollama + qwen2.5:3b). Run them with: py -m pytest -m live"""

import pytest

from ai_basics.experiments import next_token_table, repeat_answers
from ai_basics.llm import OllamaClient

pytestmark = pytest.mark.live  # marks every test in this file


@pytest.fixture
def client():
    return OllamaClient()


def test_top_next_token_after_france_is_paris(client):
    table = next_token_table(client, "The capital of France is")

    assert "Paris" in table[0][0]  # the token is probably " Paris", with a space


def test_temperature_zero_repeats_itself(client):
    answers = repeat_answers(client, "Suggest one name for a pet turtle.", n=3, temperature=0.0)

    assert len(set(answers)) == 1
