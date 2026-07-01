from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall

dataset = EvaluationDataset(goldens=[

    # golden0
    Golden(
        input="show me student 1",
        expected_output="Yash Rao is in CSE, Year 1, CGPA 7.0.",
        expected_tools=[ToolCall(name="get_student_profile",
                                 input={"student_id": 1, "detail_level": "summary"})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "faithfulness", "task_completion"]}
    ),
    # golden1
    Golden(
        input="average CGPA for computer science branch",
        expected_output="The average CGPA for CSE is 6.93.",
        expected_tools=[ToolCall(name="get_branch_stats",
                                 input={"branch": "CSE"})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "faithfulness", "task_completion"]}
    ),
    # golden2
    Golden(
        input="what is student 2's email?",
        expected_output="Email is not stored in the system.",
        expected_tools=[],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "task_completion"]}
    ),
    # golden3
    Golden(
        input="list all ECE students",
        expected_output="A list of ECE students with their names and CGPAs.",
        expected_tools=[ToolCall(name="count_or_list_students",
                                 input={"branch": "ECE", "return_list": True})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "task_completion"]}
    ),
    # golden4
    Golden(
        input="what is student 5's CGPA?",
        expected_output="Nikhil Patel's CGPA is 7.5.",
        expected_tools=[ToolCall(name="get_student_profile",
                                 input={"student_id": 5, "detail_level": "cgpa"})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "faithfulness", "task_completion"]}
    ),
    # golden5
    Golden(
        input="what is student 8's address?",
        expected_output="Kiran Chauhan's address is Plot 197, Kondapur, Hyderabad - 500044.",
        expected_tools=[ToolCall(name="get_student_profile",
                                 input={"student_id": 8, "detail_level": "basic_info"})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "faithfulness", "task_completion"]}
    ),
    # golden6
    Golden(
        input="show me audit logs",
        expected_output="A list of recent audit log entries showing who changed what and when.",
        expected_tools=[ToolCall(name="get_audit_logs",
                                 input={"student_id": None, "action": None})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "task_completion"]}
    ),
    # golden7
    Golden(
        input="show semester 1 subjects for student 3",
        expected_output="Santosh Pillai's semester 1 subjects with marks and grades.",
        expected_tools=[ToolCall(name="get_student_academic_records",
                                 input={"student_id": 3, "semester": 1})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "faithfulness", "task_completion"]}
    ),
    # golden8 — faithfulness only
    Golden(
        input="what is student 3's address?",
        expected_output="Santosh Pillai's address is Plot 12, Miyapur, Hyderabad - 500059.",
        additional_metadata={"role": "Admin", "metrics": ["faithfulness"]}
    ),
    # golden9 — faithfulness only
    Golden(
        input="what is student 1's CGPA?",
        expected_output="Yash Rao's CGPA is 7.0.",
        additional_metadata={"role": "Admin", "metrics": ["faithfulness"]}
    ),
    # golden10 — faithfulness/task_completion
    Golden(
        input="what is student 2's phone number?",
        expected_output="Phone number is not stored in the system.",
        additional_metadata={"role": "Admin", "metrics": ["faithfulness", "task_completion"]}
    ),
    # golden11 — tool_accuracy/faithfulness/task_completion
    Golden(
        input="average CGPA for mechanical branch",
        expected_output="The average CGPA for the ME branch (Mechanical) is 6.96.",
        expected_tools=[ToolCall(name="get_branch_stats",
                                 input={"branch": "ME"})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "faithfulness", "task_completion"]}
    ),
    # golden12 — tool_accuracy/faithfulness/task_completion
    Golden(
        input="how many students are in CSE?",
        expected_output="There are 120 students in CSE.",
        expected_tools=[ToolCall(name="count_or_list_students",
                                 input={"branch": "CSE", "return_list": False})],
        additional_metadata={"role": "Admin", "metrics": ["tool_accuracy", "faithfulness", "task_completion"]}
    ),
])