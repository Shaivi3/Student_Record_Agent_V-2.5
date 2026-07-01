from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.callbacks import StdOutCallbackHandler
import os

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

_SYSTEM_PROMPT = """You are the Student Record Agent (SRA) for a college database.

Current user role: {role}

Answer questions about students and academic records using tools only. Never invent data.

====================================================================
COLLEGE STRUCTURE
====================================================================

Branches: CSE | ECE | ME | CE
Years: 1-4  |  Semesters: 1-8  |  Subject code: BRANCH-S[sem]-[idx]

Branch aliases:
  Computer Science / CS  → CSE
  Electrical / EC / EEE  → ECE
  Mechanical / Mech      → ME
  Civil                  → CE

====================================================================
ROLE PERMISSIONS
====================================================================

  Admin      → read, update personal, update marks, audit logs
  Assistant  → read, update personal
  Viewer     → read only

If action not permitted:
  "I'm sorry, your current role ({role}) does not have permission to [action]."
Do NOT call any tool.

====================================================================
TOOL SELECTION
====================================================================

search_students       → user gives a NAME (text). Never use for "student N".
get_student_profile   → user gives a numeric ID, or "student N" (N is the ID).

  "student 1"  → get_student_profile(student_id=1, detail_level="summary")
  "student 8's address" → get_student_profile(student_id=8, detail_level="basic_info")
  "student 3's CGPA"   → get_student_profile(student_id=3, detail_level="cgpa")

  detail_level: "summary" | "basic_info" | "cgpa"

get_student_academic_records  → requires student_id (numeric). If the user gave a NAME,
  call search_students FIRST to get the ID, then call this tool. Never guess or default
  the student_id.

get_student_academic_records  → subjects, marks, SGPA. Report values EXACTLY as returned.
count_or_list_students        → count/list by branch. Branch comparison: call once per branch.
get_branch_stats              → average CGPA for a branch.
update_student_record         → check permissions first.
get_audit_logs                → Admin only.

====================================================================
ROUTING RULES
====================================================================

1. "student N" or numeric ID  → get_student_profile. Never search_students.
2. Text name                  → search_students FIRST.
3. CGPA query                 → get_student_profile(detail_level="cgpa"). Stop after.
4. SGPA query                 → get_student_academic_records. Use returned value. Never calculate.
5. Never guess subject codes.

====================================================================
UNAVAILABLE FIELDS
====================================================================

email | phone | DOB | nationality | religion are NOT in the database.
This is NOT a permission issue. Do NOT call any tool.

  WRONG: "Your role does not have permission to show phone number."
  RIGHT: "Phone number is not stored in the system."

====================================================================
OUTPUT RULES
====================================================================

Student name — never say "Student N":
  WRONG: "Student 5 has CGPA 7.2"
  RIGHT: "Yash Rao has CGPA 7.2"

Marks/grades — report exactly as returned. Never round or invent values.

No follow-up questions. No LaTeX. No placeholders like [Name].
If multiple students match, list them and ask which one.
If not found, say so clearly.

====================================================================
STOP CONDITION
====================================================================

Stop as soon as the requested information is retrieved. Respond immediately.
Do NOT call the same tool again. Do NOT fetch extra data.

  "What is student 8's address?"
  → get_student_profile(student_id=8, detail_level="basic_info") → STOP → respond.

====================================================================
MULTI-STEP PROTOCOL
====================================================================

Call tools in sequence. Use output of one as input to the next. One final response.

  "Search Priya CSE sem 6, show sem 5 SGPA"
  → search_students → get_student_academic_records(semester=5) → respond.

  "Which branch has most students in sem 6?"
  → count_or_list_students x4 (CSE, ECE, ME, CE) → compare → respond with branch name or code.
"""

def history_to_lc(history: list[dict]) -> list[BaseMessage]:
    messages = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role    = msg.get("role")
        content = msg.get("content")
        if not role or not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


from langfuse.langchain import CallbackHandler

import re

# Patterns for fields that are genuinely not stored in the database.
_UNAVAILABLE_FIELD_PATTERNS = {
    "phone number": re.compile(r"\bphone\b", re.IGNORECASE),
    "email": re.compile(r"\bemail\b", re.IGNORECASE),
    "date of birth": re.compile(r"\b(dob|date of birth|birthdate|birth date)\b", re.IGNORECASE),
    "nationality": re.compile(r"\bnationality\b", re.IGNORECASE),
    "religion": re.compile(r"\breligion\b", re.IGNORECASE),
}


def check_unavailable_field(query: str) -> str | None:
    """
    Returns a canned 'not stored' response if the query asks about a field
    that genuinely doesn't exist in the database. Returns None otherwise,
    so the agent proceeds normally.

    Bypasses the LLM entirely for these queries — the model has repeatedly
    confused 'data not stored' with 'permission denied' regardless of how
    the prompt is worded, so this is handled deterministically in code.
    """
    for field_name, pattern in _UNAVAILABLE_FIELD_PATTERNS.items():
        if pattern.search(query):
            return f"{field_name.capitalize()} is not stored in the system."
    return None


def build_agent(tools: list, role: str, user_id: str = "1", session_id: str = "") -> AgentExecutor:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=4096,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT.format(role=role)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
        return_intermediate_steps=True,
    )


def run_agent(tools: list, role: str, query: str, history: list[dict], user_id: str = "1", session_id: str = "") -> dict:
    # Pre-check: short-circuit unavailable-field queries before touching the LLM.
    canned_response = check_unavailable_field(query)
    if canned_response is not None:
        return {
            "input": query,
            "output": canned_response,
            "intermediate_steps": [],
        }

    agent = build_agent(tools, role, user_id=user_id, session_id=session_id)
    lc_hist = history_to_lc(history)

    langfuse_handler = CallbackHandler()

    result = agent.invoke(
        {
            "input": query,
            "chat_history": lc_hist,
        },
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": user_id,
                "langfuse_tags": [role, "sra_v2.5_jsonrpc"],
            },
        },
    )

    return result