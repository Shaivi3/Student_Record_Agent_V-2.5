import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric
from conftest import judge
from dataset import dataset
from agent_runner import run_and_capture

# Tool descriptions are critical: the judge LLM uses them to evaluate whether
# the agent picked the right tool. Without descriptions, a small judge model
# (llama3.1:8b) confuses tools — e.g. it suggested count_or_list_students for
# "show semester 1 subjects", scoring get_student_academic_records at 0.5.
# With descriptions, the judge understands each tool's purpose and scores correctly.
ALL_SYSTEM_TOOLS = [
    ToolCall(
        name="search_students",
        description="Search for students by name and optional filters (branch, year, semester). Use when the user provides a student's name instead of an ID.",
    ),
    ToolCall(
        name="get_student_profile",
        description="Retrieve a single student's profile by numeric student ID. Returns personal info, branch, year, and/or CGPA depending on detail_level (summary, basic_info, cgpa).",
    ),
    ToolCall(
        name="get_student_academic_records",
        description="Retrieve a specific student's semester-level academic records: list of subjects, marks, grades, grade points, and semester GPA. Requires student_id and semester number.",
    ),
    ToolCall(
        name="get_subject_roster_and_stats",
        description="Retrieve class-wide statistics and the full roster of students enrolled in a specific subject, identified by subject code.",
    ),
    ToolCall(
        name="count_or_list_students",
        description="Count or list all students filtered by branch, year, or semester. Returns totals and optionally the student list with CGPAs. Does NOT return subject-level or semester academic records.",
    ),
    ToolCall(
        name="get_branch_stats",
        description="Retrieve aggregate statistics for an entire branch, including the average CGPA and total student count.",
    ),
    ToolCall(
        name="update_student_record",
        description="Update a student's personal details or academic marks/grades. Requires appropriate role permissions (Admin for marks, Admin/Assistant for personal info).",
    ),
    ToolCall(
        name="get_audit_logs",
        description="Retrieve the audit log of all changes made to student records, showing who changed what and when. Admin only.",
    ),
]

tool_cases = [g for g in dataset.goldens
              if "tool_accuracy" in g.additional_metadata.get("metrics", [])]

@pytest.mark.parametrize("golden", tool_cases)
def test_tool_accuracy(golden):
    role = golden.additional_metadata.get("role", "Admin")
    captured = run_and_capture(golden.input, role)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=captured["output"],
        expected_tools=golden.expected_tools,
        tools_called=captured["tools_called"],
    )

    metric = ToolCorrectnessMetric(
        available_tools=ALL_SYSTEM_TOOLS,
        should_consider_ordering=False,
        threshold=0.7,
        model=judge,
        include_reason=True,
    )

    assert_test(test_case, [metric], run_async=False)