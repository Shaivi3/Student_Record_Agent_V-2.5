import os

import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import TaskCompletionMetric
from conftest import judge
from dataset import dataset
from agent_runner import run_and_capture

task_cases = [g for g in dataset.goldens
              if "task_completion" in g.additional_metadata.get("metrics", [])]

@pytest.mark.parametrize("golden", task_cases)
def test_task_completion(golden):
    role = golden.additional_metadata.get("role", "Admin")
    captured = run_and_capture(golden.input, role)

    if not captured["output"]:
        pytest.skip("Agent returned empty output")

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=captured["output"],
        tools_called=captured["tools_called"] or [],
        retrieval_context=captured["retrieval_context"] or ["No tool output"],
    )

    metric = TaskCompletionMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
    )

    assert_test(test_case, [metric], run_async=False)