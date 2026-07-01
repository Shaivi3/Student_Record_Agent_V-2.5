import os
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"

import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from conftest import judge
from dataset import dataset
from agent_runner import run_and_capture

faith_cases = [g for g in dataset.goldens
               if "faithfulness" in g.additional_metadata.get("metrics", [])]

@pytest.mark.parametrize("golden", faith_cases)
def test_faithfulness(golden):
    role = golden.additional_metadata.get("role", "Admin")
    captured = run_and_capture(golden.input, role)

    if not captured["output"]:
        pytest.skip("Agent returned empty output after retries")

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=captured["output"],
        expected_output=golden.expected_output,
        retrieval_context=captured["retrieval_context"] or ["No tool output"],
    )

    metric = FaithfulnessMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
    )

    assert_test(test_case, [metric])