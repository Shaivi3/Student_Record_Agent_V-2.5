"""
srs_main.py — Student Record Service (SRS), v2.5.
Owns the database. Runs on port 8001.

v2.5 change: REST routes collapsed into a single JSON-RPC 2.0 endpoint (/rpc).
All business logic functions are unchanged from v2 — only the transport layer
changed from per-resource REST routes to method-dispatched JSON-RPC.
"""
from dotenv import load_dotenv
load_dotenv()

import re
from typing import Optional, Literal, Any
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from generate_students import (
    SessionLocal, Student, Subject, Enrollment, Mark, SemesterGPA,
    get_student_by_id, search_students_by_name, update_personal,
    query_audit_logs, list_branch_subjects, get_student_subjects,
    get_student_subject_mark, get_branch_average_cgpa as get_branch_average_cgpa_query,
    get_students_by_branch_or_year as get_students_by_branch_or_year_query,
    count_students as count_students_query, update_marks as update_marks_query,
)

app = FastAPI(title="Student Record Service (SRS) v2.5 — JSON-RPC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _subject_semester(subject_code: str) -> Optional[int]:
    match = re.search(r"-s(\d{2})-", subject_code.lower())
    return int(match.group(1)) if match else None


def _grade_point_for_grade(grade: str) -> Optional[float]:
    return {
        "A+": 10.0, "A": 9.0, "B": 8.0, "C": 7.0,
        "D": 6.0, "E": 5.0, "F": 0.0,
    }.get(grade.upper())


def _subject_rows(db: Session, subject_code: str, active_only: bool = True):
    subject = db.query(Subject).filter(Subject.code == subject_code).first()
    if not subject:
        return None, []

    query = (
        db.query(Student, Mark)
        .join(Enrollment, Enrollment.student_id == Student.id)
        .join(Mark, Mark.enrollment_id == Enrollment.id)
        .filter(Enrollment.subject_id == subject.id)
    )

    if active_only:
        subject_year = (subject.semester + 1) // 2
        query = query.filter(Student.year == subject_year)

    rows = query.order_by(Student.id, Mark.recorded_at.desc(), Mark.id.desc()).all()
    latest_by_student = {}
    for student, mark in rows:
        latest_by_student.setdefault(student.id, (student, mark))
    return subject, list(latest_by_student.values())


# ── Business logic functions (unchanged from v2, just decoupled from FastAPI routing) ──

def logic_search_students(db: Session, name: Optional[str] = None, branch: Optional[str] = None,
                           semester: Optional[int] = None, limit: int = 10):
    matches = search_students_by_name(db, name.strip() if name else "", limit)
    if branch:
        matches = [r for r in matches if r["branch"].upper() == branch.upper()]
    if semester is not None:
        matches = [r for r in matches if r["current_semester"] == semester]
    if not matches:
        return {"status": "NOT_FOUND", "message": f"No students found matching '{name}'."}
    return {"status": "OK", "results": matches}


def logic_get_student(db: Session, student_id: int):
    res = get_student_by_id(db, student_id)
    if res is None:
        return {"status": "NOT_FOUND", "message": f"No student found with id {student_id}"}
    return {"status": "OK", "student": res}


def logic_count_or_list_students(db: Session, year: Optional[int] = None, semester: Optional[int] = None,
                                  branch: Optional[str] = None, return_list: bool = False):
    if not return_list:
        count = count_students_query(db, year, semester, branch)
        return {"status": "OK", "count": count}

    raw = get_students_by_branch_or_year_query(db, branch=branch, year=year) or []
    if semester is not None:
        raw = [s for s in raw if s.get("current_semester") == semester]
    if not raw:
        return {"status": "NOT_FOUND", "message": "No students found matching those criteria."}
    return {
        "status": "OK",
        "total": len(raw),
        "students": [
            {"student_id": s["student_id"], "name": s["name"], "cgpa": s.get("cgpa")}
            for s in raw
        ],
    }


def logic_count_all_branches(db: Session, semester: int):
    counts = {b: count_students_query(db, semester=semester, branch=b)
              for b in ("CSE", "ECE", "ME", "CE")}
    highest = max(counts.values())
    return {
        "status": "OK", "semester": semester, "counts": counts,
        "highest_count": highest,
        "highest_branches": [b for b, c in counts.items() if c == highest],
    }


def logic_search_subjects(db: Session, name: str):
    rows = db.query(Subject).filter(Subject.name.ilike(f"%{name}%")).all()
    if not rows:
        return {"status": "NOT_FOUND", "message": f"No subject found matching '{name}'."}
    return {
        "status": "OK",
        "subjects": [
            {"subject_code": r.code, "subject_name": r.name,
             "branch": r.branch, "semester": r.semester}
            for r in rows
        ]
    }


def logic_list_semester_subjects(db: Session, branch: str, semester: int):
    res = list_branch_subjects(db, branch, semester)
    if not res:
        return {"status": "NOT_FOUND", "message": f"No subjects found for {branch} semester {semester}"}
    return {"status": "OK", "subjects": res}


def logic_get_subject_roster_and_stats(db: Session, subject_code: Optional[str] = None,
                                        name: Optional[str] = None, include_roster: bool = False,
                                        all_years: bool = False):
    if not subject_code and name:
        rows = db.query(Subject).filter(Subject.name.ilike(f"%{name}%")).all()
        if not rows:
            return {"status": "NOT_FOUND", "message": f"No subject found matching '{name}'."}
        if len(rows) == 1:
            subject_code = rows[0].code
        else:
            return {
                "status": "MULTIPLE_FOUND",
                "subjects": [
                    {"subject_code": s.code, "subject_name": s.name,
                     "branch": s.branch, "semester": s.semester}
                    for s in rows
                ],
                "message": "Multiple subjects found. Call again with the correct subject_code."
            }

    if not subject_code:
        return {"status": "ERROR", "message": "Provide either subject_code or name."}

    active_only = not all_years
    subject, rows = _subject_rows(db, subject_code, active_only)

    if not subject:
        matched = db.query(Subject).filter(
            Subject.name.ilike(f"%{subject_code.strip()}%")
        ).first()
        if matched:
            subject, rows = _subject_rows(db, matched.code, active_only)
            subject_code = matched.code

    if not subject:
        return {"status": "NOT_FOUND", "message": f"No subject found matching '{subject_code}'."}

    stats = {
        "status": "OK",
        "subject_code": subject_code,
        "subject_name": subject.name,
        "student_count": len(rows),
    }

    if not rows:
        return {**stats, "average_marks": None, "highest_marks": None,
                "highest_scorers": [], "lowest_marks": None, "pass_percentage": None}

    marks_list = [m.marks for _, m in rows]
    highest = max(marks_list)
    pass_count = sum(1 for _, m in rows if (m.grade or "").upper() != "F")

    stats.update({
        "average_marks": round(sum(marks_list) / len(rows), 2),
        "highest_marks": highest,
        "highest_scorers": [
            {"student_id": s.id, "name": s.name}
            for s, m in rows if m.marks == highest
        ],
        "lowest_marks": min(marks_list),
        "pass_percentage": round(pass_count / len(rows) * 100, 2),
    })

    if include_roster:
        stats["roster"] = [
            {"student_id": s.id, "name": s.name, "marks": m.marks,
             "grade": m.grade, "grade_point": m.grade_point}
            for s, m in rows
        ]

    return stats


def logic_get_branch_average_cgpa(db: Session, branch: str):
    res = get_branch_average_cgpa_query(db, branch)
    if res is None:
        return {"status": "NOT_FOUND", "message": f"No data found for branch {branch}."}
    if isinstance(res, dict):
        return {"status": "OK", **res}
    return {"status": "OK", "branch": branch, "average_cgpa": res}


def logic_update_student_record(db: Session, student_id: int, update_type: str,
                                 payload: dict, user_id: int, role: str):
    if update_type == 'personal':
        updated = update_personal(db, user_id, role, student_id, payload.get("fields", {}))
        if updated is None:
            return {"status": "NOT_FOUND", "message": f"Student ID {student_id} not found."}
        return {"status": "OK", "updated": updated}

    if update_type == 'marks':
        sem = payload.get("semester")
        sub_code = payload.get("subject_code")
        marks = payload.get("marks")
        m = int(marks)
        if m >= 90:   g, gp = "A+", 10.0
        elif m >= 80: g, gp = "A",  9.0
        elif m >= 70: g, gp = "B",  8.0
        elif m >= 60: g, gp = "C",  7.0
        elif m >= 50: g, gp = "D",  6.0
        elif m >= 40: g, gp = "E",  5.0
        else:         g, gp = "F",  0.0
        res = update_marks_query(db, user_id, {
            "student_id": student_id, "semester": sem, "subject_code": sub_code,
            "marks": marks, "grade": g, "grade_point": gp
        })
        if res is None:
            return {"status": "NOT_FOUND", "message": "Student or subject code not found."}
        return {"status": "OK", "new_record": res}

    if update_type == 'grade':
        sub_code = payload.get("subject_code")
        grade = payload.get("grade").upper()
        sem = _subject_semester(sub_code)
        gp = _grade_point_for_grade(grade)
        if sem is None or gp is None:
            return {"status": "ERROR", "message": "Invalid subject code or grade."}
        existing = get_student_subject_mark(db, student_id, sub_code)
        if existing is None:
            return {"status": "NOT_FOUND", "message": "Student has no record for that subject."}
        before = {s["subject_code"]: s for s in get_student_subjects(db, student_id, sem)}
        res = update_marks_query(db, user_id, {
            "student_id": student_id, "semester": sem, "subject_code": sub_code,
            "marks": existing["marks"], "grade": grade, "grade_point": gp
        })
        after = {s["subject_code"]: s for s in get_student_subjects(db, student_id, sem)}
        changed = [
            {"subject_code": code, "old_grade": before[code]["grade"],
             "new_grade": after[code]["grade"]}
            for code in after
            if code in before and before[code]["grade"] != after[code]["grade"]
        ]
        return {"status": "OK", "new_record": res, "changed_subjects": changed}

    return {"status": "ERROR", "message": f"Unknown update_type: {update_type}"}


def logic_get_audit_logs(db: Session, student_id: Optional[int] = None,
                          action: Optional[str] = None, limit: int = 50):
    entries = query_audit_logs(db, student_id=student_id, action=action, limit=limit)
    return {"status": "OK", "entries": entries}


# ── JSON-RPC 2.0 dispatcher ─────────────────────────────────────────────────

class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict = {}
    id: Optional[Any] = None


# Method registry: rpc method name -> (logic function, requires_db_first_arg)
# Every logic_* function takes db as its first positional argument; the rest
# are passed as **params straight from the JSON-RPC request body.
METHODS = {
    "students.search":            logic_search_students,
    "students.profile":           logic_get_student,
    "students.count_or_list":     logic_count_or_list_students,
    "students.count_all_branches": logic_count_all_branches,
    "subjects.search":            logic_search_subjects,
    "subjects.list":               logic_list_semester_subjects,
    "subjects.roster_and_stats":  logic_get_subject_roster_and_stats,
    "branches.average_cgpa":      logic_get_branch_average_cgpa,
    "records.update":             logic_update_student_record,
    "audit.logs":                 logic_get_audit_logs,
}

# JSON-RPC 2.0 standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _rpc_error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _rpc_result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


@app.post("/rpc")
def rpc_dispatch(req: JsonRpcRequest, db: Session = Depends(get_db)):
    if req.jsonrpc != "2.0":
        return _rpc_error(req.id, INVALID_REQUEST, "jsonrpc must be '2.0'")

    fn = METHODS.get(req.method)
    if fn is None:
        return _rpc_error(req.id, METHOD_NOT_FOUND, f"Unknown method: {req.method}")

    try:
        result = fn(db, **req.params)
        return _rpc_result(req.id, result)
    except TypeError as e:
        return _rpc_error(req.id, INVALID_PARAMS, f"Invalid params for {req.method}: {e}")
    except Exception as e:
        return _rpc_error(req.id, INTERNAL_ERROR, str(e))


@app.get("/rpc/methods")
def list_methods():
    """Convenience endpoint (not part of JSON-RPC spec) to discover available methods."""
    return {"methods": list(METHODS.keys())}