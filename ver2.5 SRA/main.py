"""
main.py — FastAPI entrypoint for the Student Record Agent (Version 2.5).
Routes all tool interactions to the Student Record Service (SRS) over JSON-RPC 2.0
(single POST /rpc endpoint on SRS), instead of v2's per-resource REST calls.
Tool definitions and behavior are otherwise unchanged from v2.
"""
from dotenv import load_dotenv
load_dotenv()

import json
import logging
from typing import Optional, Literal

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain_core.tools import tool
import requests

from Auth import AuthUser, LoginRequest, create_token, authenticate, get_current_user
from Agent import run_agent
from generate_students import (
    SessionLocal, init_db,
    create_chat_session, list_chat_sessions, load_chat_session_history,
    save_chat_exchange, rename_chat_session, delete_chat_session,
    save_conversation, load_conversation,
)

logger = logging.getLogger("sra")

app = FastAPI(title="Student Record Agent v2.5 (JSON-RPC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SRS_URL = "http://127.0.0.1:8001"
SRS_RPC_URL = f"{SRS_URL}/rpc"

_rpc_id_counter = 0


def srs_rpc(method: str, **params) -> dict:
    """
    Call the SRS over JSON-RPC 2.0. Strips None-valued params before sending
    (mirrors the old behavior where requests.get silently dropped None query params).
    Always returns the inner application payload dict (e.g. {"status": "OK", ...}),
    same shape tools previously got back from res.json() over REST.
    """
    global _rpc_id_counter
    _rpc_id_counter += 1
    clean_params = {k: v for k, v in params.items() if v is not None}
    envelope = {
        "jsonrpc": "2.0",
        "method": method,
        "params": clean_params,
        "id": _rpc_id_counter,
    }
    try:
        res = requests.post(SRS_RPC_URL, json=envelope, timeout=15)
        body = res.json()
    except Exception as e:
        return {"status": "ERROR", "message": f"RPC transport failure: {e}"}

    if body.get("error") is not None:
        err = body["error"]
        return {"status": "ERROR", "message": err.get("message", "RPC error"),
                "rpc_code": err.get("code")}

    return body.get("result", {"status": "ERROR", "message": "Malformed RPC response"})


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    session_id: Optional[str] = None
    title: Optional[str] = None
    history: Optional[list[dict]] = None


class ChatSessionRequest(BaseModel):
    session_id: str
    title: Optional[str] = "New chat"


class RenameSessionRequest(BaseModel):
    title: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup_event():
    init_db()


def build_tools(db: Session, role: str, user_id: int):

    @tool
    def search_students(name: str, branch: Optional[str] = None,
                        semester: Optional[int] = None, limit: int = 10) -> dict:
        """Find students by name (partial match). Use whenever the user gives a name instead of an ID.
        Call this IMMEDIATELY when a name appears in the query — do NOT ask for clarification first.

        branch: Optional filter. Convert user input to code before passing:
            CSE = "Computer Science", "CS", "Comp Sci", "CSE"
            ECE = "Electrical", "Electronics", "EC", "EEE", "ECE"
            ME  = "Mechanical", "Mech", "Mech Eng", "ME"
            CE  = "Civil", "Civil Eng", "CE"

        semester: Optional int 1-8.

        If multiple matches are returned, list them and ask the user which one they mean.
        Do NOT proceed with a guess. NEVER proceed with an update if student identity is ambiguous.

        Examples:
            "find Priya in CSE semester 6" -> search_students(name="Priya", branch="CSE", semester=6)
            "search for Rahul"             -> search_students(name="Rahul")
            "who is Bhavana Pandey?"       -> search_students(name="Bhavana Pandey")
        """
        return srs_rpc("students.search", name=name, branch=branch,
                       semester=semester, limit=limit)

    @tool
    def get_student_profile(student_id: int,
                            detail_level: Literal['basic_info', 'summary', 'cgpa', 'full_history']) -> dict:
        """Get a student's profile by numeric ID. Call IMMEDIATELY when a numeric ID is given.

        detail_level — choose one:
            'basic_info'   — Returns name, gender, father_name, address, branch, year.
                             USE THIS for: "address", "where does he live", "father name",
                             "who is student X", "basic details".
                             name, gender, father_name, address ARE stored and available.
            'summary'      — CGPA + all semester SGPAs. Use for "summarize", "performance overview",
                             "show details", "tell me about student X".
            'cgpa'         — CGPA and current_semester only. Use for "what is X's CGPA".
            'full_history' — Every semester, every subject, marks, grades.
                             Use ONLY when user asks for ALL semesters or complete history.

        STORED IN DB: name, gender, father_name, address, branch, year, CGPA, marks, grades.
        NOT stored (say unavailable, do not call tools): email, phone, DOB, nationality, religion.

        Examples:
            "show details for student 5"    -> get_student_profile(student_id=5, detail_level='summary')
            "what is student 10's CGPA?"    -> get_student_profile(student_id=10, detail_level='cgpa')
            "address of student 8"          -> get_student_profile(student_id=8, detail_level='basic_info')
            "full academic history of ID 3" -> get_student_profile(student_id=3, detail_level='full_history')
        """
        try:
            s_data = srs_rpc("students.profile", student_id=student_id)
            if s_data.get("status") != "OK":
                return s_data

            student = s_data["student"]

            if detail_level == 'cgpa':
                return {"status": "OK", "id": student["id"], "name": student["name"],
                        "current_semester": student["current_semester"], "cgpa": student["cgpa"]}

            if detail_level == 'basic_info':
                return {"status": "OK", "student": {
                    k: student[k] for k in
                    ["id", "name", "gender", "father_name", "address", "branch", "year"]
                    if k in student
                }}

            if detail_level == 'summary':
                return {"status": "OK", "student": {
                    "id": student["id"], "name": student["name"], "branch": student["branch"],
                    "year": student["year"], "current_semester": student["current_semester"],
                    "cgpa": student["cgpa"],
                    "semester_gpas": [{"semester": s["semester"], "sgpa": s["gpa"]}
                                      for s in student.get("semesters", [])],
                }}

            return s_data
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    @tool
    def get_student_academic_records(student_id: int, semester: int,
                                      subject_code: Optional[str] = None) -> dict:
        """Get academic records for ONE student in ONE semester (1-8).

        subject_code optional:
            Omit    -> returns all subjects + marks + grades + SGPA for that semester.
                       Use the returned 'semester_gpa' field for SGPA. Never calculate manually.
            Provide -> returns marks/grade/grade_point for that one subject only.

        Subject code format: BRANCH-S[sem]-[idx]  e.g. CSE-S05-01, ECE-S03-02

        Examples:
            "show semester 2 subjects for student 10" -> get_student_academic_records(student_id=10, semester=2)
            "SGPA of student 10 in semester 2"        -> get_student_academic_records(student_id=10, semester=2)
            "what did student 3 get in CSE-S01-02?"   -> get_student_academic_records(student_id=3, semester=1, subject_code="CSE-S01-02")
        """
        try:
            s_data = srs_rpc("students.profile", student_id=student_id)
            if s_data.get("status") != "OK":
                return s_data

            sem_data = next((s for s in s_data["student"].get("semesters", [])
                             if s["semester"] == semester), None)
            if not sem_data:
                return {"status": "NOT_FOUND", "message": f"No records found for semester {semester}."}

            if subject_code:
                subj = next((s for s in sem_data.get("subjects", [])
                             if s["subject_code"].upper() == subject_code.upper()), None)
                if not subj:
                    return {"status": "NOT_FOUND",
                            "message": f"Subject {subject_code} not found in semester {semester}."}
                return {"status": "OK", **subj}

            return {
                "status": "OK",
                "student_id": student_id,
                "semester": semester,
                "semester_gpa": sem_data.get("gpa"),
                "subjects": sem_data.get("subjects", []),
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    @tool
    def get_subject_roster_and_stats(subject_code: Optional[str] = None,
                                      name: Optional[str] = None,
                                      include_roster: bool = False,
                                      all_years: bool = False) -> dict:
        """Get statistics and optionally the full student roster for a subject.

        Identify the subject by code OR by name — provide at least one.
        NEVER guess a subject code. If user gives a name, pass it as name= and let this tool resolve it.

        subject_code: Exact format BRANCH-S[sem]-[idx], e.g. CSE-S05-04.
        name: Subject name or partial name, e.g. "Earthquake Engineering", "Machine Learning".
              If name matches multiple subjects, all matches returned — call again with subject_code.

        include_roster:
            False (default) -> stats only: average, highest, lowest, pass%, topper(s).
            True            -> stats + full list of every enrolled student with marks and grade.

        all_years:
            False (default) -> current enrolled students only.
            True            -> all students who ever took this subject across all years.
                               Use only when user explicitly asks for "all years" or "historical".

        Examples:
            "average marks in Earthquake Engineering"  -> get_subject_roster_and_stats(name="Earthquake Engineering")
            "class stats for CSE-S05-04"               -> get_subject_roster_and_stats(subject_code="CSE-S05-04")
            "list all students in CSE-S05-04"          -> get_subject_roster_and_stats(subject_code="CSE-S05-04", include_roster=True)
        """
        return srs_rpc("subjects.roster_and_stats", subject_code=subject_code, name=name,
                       include_roster=include_roster, all_years=all_years)

    @tool
    def count_or_list_students(branch: Optional[str] = None, year: Optional[int] = None,
                                semester: Optional[int] = None,
                                return_list: bool = False) -> dict:
        """Count or list students filtered by branch, year, and/or semester.

        branch: Convert user input to code before passing:
            CSE = "Computer Science", "CS", "Comp Sci"
            ECE = "Electrical", "Electronics", "EC", "EEE"
            ME  = "Mechanical", "Mech", "Mech Eng"
            CE  = "Civil", "Civil Eng"

        return_list=False -> count only. Use for "how many students in CSE?"
        return_list=True  -> count + list of names + cgpa per student.

        To compare branches (e.g. "which branch has most students in sem 6"):
            Call this tool FOUR times, once per branch, with the same semester.
            Then compare the counts and state the result.

        Do NOT pass year and semester together unless user explicitly asks for both.

        Examples:
            "how many students in CSE?"            -> count_or_list_students(branch="CSE", return_list=False)
            "list 1st year ECE students"           -> count_or_list_students(branch="ECE", year=1, return_list=True)
            "which branch has most in semester 6?" -> call four times with semester=6, each branch
        """
        return srs_rpc("students.count_or_list", year=year, semester=semester,
                       branch=branch, return_list=return_list)

    @tool
    def get_branch_stats(branch: str) -> dict:
        """Get average CGPA and total student count for an entire branch.
        branch codes: CSE, ECE, ME, CE (convert aliases before calling).

        Use IMMEDIATELY for: "average CGPA for CSE", "how is ME branch performing overall".
        Do NOT call count_or_list_students in a loop to calculate averages — use this instead.

        Examples:
            "average CGPA for CSE students"   -> get_branch_stats(branch="CSE")
            "what is the ECE branch average?" -> get_branch_stats(branch="ECE")
        """
        return srs_rpc("branches.average_cgpa", branch=branch)

    @tool
    def update_student_record(student_id: int,
                               update_type: Literal['personal', 'marks', 'grade'],
                               payload: dict) -> dict:
        """Modify a student record. Check role permissions before calling.

        ROLE PERMISSIONS:
            personal updates: Admin or Assistant only.
            marks updates: Admin only.
            grade updates: Admin only.

        update_type and payload formats:
            personal: Update name, gender, father_name, or address.
                payload = {"fields": {"name": "New Name", "address": "New Addr"}}

            marks: Update numeric score. Grade auto-calculates from marks.
                payload = {"semester": 5, "subject_code": "CSE-S05-01", "marks": 95.0}

            grade: Change letter grade only. Recalculates GPA and CGPA automatically.
                payload = {"subject_code": "CSE-S05-01", "grade": "C"}
        """
        if update_type == 'personal' and role not in ("Admin", "Assistant"):
            return {"status": "DENIED", "message": "Only Admin or Assistant can update personal details."}
        if update_type in ('marks', 'grade') and role != "Admin":
            return {"status": "DENIED", "message": "Only Admins can modify academic records."}
        return srs_rpc("records.update", student_id=student_id, update_type=update_type,
                       payload=payload, user_id=user_id, role=role)

    @tool
    def get_audit_logs(student_id: Optional[int] = None, action: Optional[str] = None,
                       limit: int = 50) -> dict:
        """CRITICAL ADMINISTRATIVE TOOL: View the audit trail of modifications, updates, and changes made to student records. Admin only.

        CRITICAL SELECTION RULE: 
        - ONLY call this tool if the user explicitly uses words like: "audit", "log", "history of changes", "who changed", or "modifications made".
        - NEVER call this tool automatically after counting, searching, or listing students.
        - If the user asks to "list all students", "how many students", or "show roster", calling this tool is COMPLETELY WRONG and an error.

        Call once per query. Do not re-call with a student_id you discovered from a previous result
        
        student_id: Optional integer. Filter to ONE specific student's change history. Leave as None if the user is asking about general system changes.
        action: Optional string. Filter by action type ('update_marks', 'update_personal', 'create_student'). Leave as None unless specified.
        limit: Max entries to return. Default is 50.

        If no audit entries exist, say "No changes have been recorded."
        """
        if role != "Admin":
            return {"status": "DENIED", "message": "Only Admins can view audit logs."}
        return srs_rpc("audit.logs", student_id=student_id, action=action, limit=limit)

    return [
        search_students, get_student_profile, get_student_academic_records,
        get_subject_roster_and_stats, count_or_list_students, get_branch_stats,
        update_student_record, get_audit_logs,
    ]


# ── Auth ──────────────────────────────────────

@app.post("/auth/login")
def login(payload: LoginRequest):
    user = authenticate(payload.username, payload.password)
    token = create_token(payload.username)
    return {"access_token": token, "token_type": "bearer",
            "role": user["role"], "user_id": user["user_id"]}


@app.post("/auth/token")
def login_swagger(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form_data.username, form_data.password)
    token = create_token(form_data.username)
    return {"access_token": token, "token_type": "bearer",
            "role": user["role"], "user_id": user["user_id"]}


# ── Chat sessions ─────────────────────────────

@app.get("/chat/sessions")
def api_list_chat_sessions(user: AuthUser = Depends(get_current_user),
                            db: Session = Depends(get_db)):
    return {"status": "OK", "sessions": list_chat_sessions(db, user.user_id)}


@app.post("/chat/sessions")
def api_create_chat_session(payload: ChatSessionRequest,
                             user: AuthUser = Depends(get_current_user),
                             db: Session = Depends(get_db)):
    session = create_chat_session(db, user.user_id, user.role,
                                   payload.session_id, payload.title or "New chat")
    return {"status": "OK", "session": session}


@app.patch("/chat/sessions/{session_id}")
def api_rename_chat_session(session_id: str, payload: RenameSessionRequest,
                             user: AuthUser = Depends(get_current_user),
                             db: Session = Depends(get_db)):
    session = rename_chat_session(db, user.user_id, session_id, payload.title)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"status": "OK", "session": session}


@app.delete("/chat/sessions/{session_id}")
def api_delete_chat_session(session_id: str,
                             user: AuthUser = Depends(get_current_user),
                             db: Session = Depends(get_db)):
    if not delete_chat_session(db, user.user_id, session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"status": "OK"}


# ── Main chat endpoint ────────────────────────

# ── Main chat endpoint (main.py) ────────────────────────

@app.post("/chat")
def chat(payload: ChatRequest, user: AuthUser = Depends(get_current_user),
         db: Session = Depends(get_db)):
    query = (payload.query or payload.message or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query or message is required")

    if payload.history is None and payload.session_id:
        history = load_chat_session_history(db, user.user_id, payload.session_id)
    elif payload.history is None:
        raw = load_conversation(db, user.user_id)
        history = json.loads(raw) if raw else []
    else:
        history = payload.history

    tools = build_tools(db, user.role, user.user_id)
    agent_failed = False
    answer = ""
    agent_result = {}
    try:
        # This securely passes telemetry down to Agent.py for dashboard reporting
        agent_result = run_agent(
            tools, user.role, query, history,
            user_id=str(user.user_id),
            session_id=payload.session_id or "",
        )
        answer = agent_result.get("output", "").strip()
    except Exception:
        logger.exception("Agent failed on query: %r", query)
        agent_failed = True

    if not answer or not answer.strip():
        answer = "Sorry, I could not process that request. Please try rephrasing it, or try again."
        agent_failed = True

    if not agent_failed:
        updated_history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        if payload.session_id:
            save_chat_exchange(db, user.user_id, user.role, payload.session_id,
                               payload.title or query[:42] or "New chat", query, answer)
        else:
            save_conversation(db, user.user_id, json.dumps(updated_history))
    else:
        updated_history = history
        logger.warning("Skipped saving chat history for user %s due to agent failure.", user.user_id)

    return {"status": "OK", "response": answer, "history": updated_history}