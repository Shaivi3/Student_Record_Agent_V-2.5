import sys
import os
import json
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from deepeval.test_case import ToolCall
from Agent import build_agent
from main import build_tools
from generate_students import SessionLocal


DEBUG_EVAL = os.getenv("DEBUG_EVAL", "0") == "1"

_DETAIL_LEVEL_ALIASES = {
    "basic_info": "summary",
    "personal": "summary",
    "cgpa": "summary",
    "full": "summary",
    "academic": "summary",
}


_NO_TOOL_CONTEXT = (
    "No database tool was called. "
    "The agent answered directly without retrieving database records."
)

# -------------------------------------------------------------------
# Tool input normalization
# -------------------------------------------------------------------

def _normalize_tool_input(tool_name: str, tool_input):

    if not isinstance(tool_input, dict):

        if tool_name == "get_student_profile":
            try:
                tool_input = {
                    "student_id": int(tool_input),
                    "detail_level": "summary",
                }
            except Exception:
                tool_input = {
                    "student_id": tool_input,
                    "detail_level": "summary",
                }

        elif tool_name == "count_or_list_students":
            tool_input = {
                "branch": str(tool_input),
                "return_list": True,
            }

        else:
            tool_input = {
                "input": tool_input
            }

    tool_input = dict(tool_input)

    if "detail_level" in tool_input:
        tool_input["detail_level"] = _DETAIL_LEVEL_ALIASES.get(
            tool_input["detail_level"],
            tool_input["detail_level"],
        )

    return tool_input


# -------------------------------------------------------------------
# Pretty formatting helpers
# -------------------------------------------------------------------

def _pretty_key(key: str) -> str:
    return key.replace("_", " ").title()


def _format_scalar(key, value):

    if value is None:
        return None

    if value == "":
        return None

    return f"{_pretty_key(key)}: {value}"


def _format_semester_gpas(values):

    lines = []

    for sgpa in values:

        semester = sgpa.get("semester")
        score = sgpa.get("sgpa")

        lines.append(
            f"Semester {semester} SGPA: {score}"
        )

    return lines


def _format_subjects(values):

    lines = []

    for subject in values:

        parts = []

        code = subject.get("subject_code")
        name = subject.get("subject_name")

        if code or name:
            parts.append(
                f"{code} ({name})"
            )

        if "marks" in subject:
            parts.append(
                f"Marks: {subject['marks']}"
            )

        if "grade" in subject:
            parts.append(
                f"Grade: {subject['grade']}"
            )

        if "grade_point" in subject:
            parts.append(
                f"Grade Point: {subject['grade_point']}"
            )

        if "credits" in subject:
            parts.append(
                f"Credits: {subject['credits']}"
            )

        lines.append(" | ".join(parts))

    return lines


# -------------------------------------------------------------------
# Generic recursive formatter
# -------------------------------------------------------------------

def _flatten(value, prefix=""):

    lines = []

    if isinstance(value, dict):

        for key, item in value.items():

            if key == "status":
                continue

            if item is None:
                continue

            if key == "semester_gpas":
                lines.extend(
                    _format_semester_gpas(item)
                )
                continue

            if key == "subjects":
                lines.extend(
                    _format_subjects(item)
                )
                continue

            next_prefix = (
                f"{prefix}{_pretty_key(key)} "
                if prefix
                else f"{_pretty_key(key)} "
            )

            if isinstance(item, (dict, list)):
                lines.extend(
                    _flatten(item, next_prefix)
                )
            else:

                text = _format_scalar(
                    f"{prefix}{key}",
                    item,
                )

                if text:
                    lines.append(text)

    elif isinstance(value, list):

        for item in value:
            lines.extend(
                _flatten(item, prefix)
            )

    else:

        if value is not None:
            lines.append(
                str(value)
            )

    return lines


def _flatten_observation(tool_name, observation):

    if isinstance(observation, str):

        try:
            observation = json.loads(observation)
        except Exception:
            return observation

    if not isinstance(observation, dict):
        return str(observation)

    if observation.get("status") != "OK":
        return observation.get(
        "message",
        str(observation)
)

    lines = [
        f"Tool: {tool_name}",
        ""
    ]

    lines.extend(
        _flatten(observation)
    )

    return "\n".join(lines)


def _deduplicate_context(chunks):

    unique = OrderedDict()

    for chunk in chunks:

        text = chunk.strip()

        if text:
            unique[text] = None

    return list(unique.keys())

def run_and_capture(
    query: str,
    role: str,
    history: list | None = None,
    retries: int = 2,
):

    if history is None:
        history = []

    db = SessionLocal()

    try:

        user_id = 1 if role == "Admin" else 2

        tools = build_tools(
            db,
            role=role,
            user_id=user_id,
        )

        output = ""
        steps = []

        # -------------------------
        # Retry loop
        # -------------------------

        for attempt in range(retries + 1):

            agent = build_agent(
                tools=tools,
                role=role,
                user_id=str(user_id),
                session_id=f"eval_session_{user_id}",
            )

            result = agent.invoke(
                {
                    "input": query,
                    "chat_history": [],
                }
            )

            output = result.get("output", "").strip()
            steps = result.get("intermediate_steps", [])

            if output:
                break

            if attempt < retries:
                print(
                    f"[Retry {attempt+1}] Empty output. Retrying..."
                )

        # ------------------------------------------------
        # Build ToolCorrectness objects (RAW JSON)
        # ------------------------------------------------

        tools_called = []

        retrieval_chunks = []

        seen_tool_calls = set()

        for action, observation in steps:

            tool_input = _normalize_tool_input(
                action.tool,
                action.tool_input,
            )

            if isinstance(observation, dict):
                obs_json = json.dumps(
                    observation,
                    ensure_ascii=False,
                    default=str,
                )
            else:
                obs_json = str(observation)

            signature = (
                action.tool,
                json.dumps(tool_input, sort_keys=True),
                obs_json,
            )

            # Avoid identical repeated tool calls
            if signature not in seen_tool_calls:

                seen_tool_calls.add(signature)

                tools_called.append(
                    ToolCall(
                        name=action.tool,
                        input=tool_input,
                        output=obs_json,
                    )
                )

            retrieval_chunks.append(
                _flatten_observation(
                    action.tool,
                    observation,
                )
            )

        # ---------------------------------------------
        # Faithfulness retrieval context
        # ---------------------------------------------

        retrieval_chunks = _deduplicate_context(
            retrieval_chunks
        )

        if retrieval_chunks:

            retrieval_context = [
                "\n\n"
                + ("\n\n" + "-" * 70 + "\n\n").join(
                    retrieval_chunks
                )
            ]

        else:

            retrieval_context = [_NO_TOOL_CONTEXT]

        # ---------------------------------------------
        # Optional debugging
        # ---------------------------------------------

        if DEBUG_EVAL:

            print("\n")
            print("=" * 80)
            print("RETRIEVAL CONTEXT")
            print("=" * 80)
            print(retrieval_context[0])
            print("=" * 80)

            print("\nTOOLS CALLED")

            for tool in tools_called:
                print(tool.name, tool.input)

            print("=" * 80)

        return {
            "output": output,
            "tools_called": tools_called,
            "retrieval_context": retrieval_context,
            "tool_was_called": len(tools_called) > 0,
        }

    finally:
        db.close()


if __name__ == "__main__":

    result = run_and_capture(
        "show me student 1",
        "Admin",
    )

    print("\nOUTPUT\n")
    print(result["output"])

    print("\nTOOLS\n")
    for tool in result["tools_called"]:
        print(tool)

    print("\nRETRIEVAL CONTEXT\n")
    print(result["retrieval_context"][0])