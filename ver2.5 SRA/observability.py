from dotenv import load_dotenv
load_dotenv()

import os
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)


def trace_agent_run(query: str, role: str, user_id: int, session_id: str, result: dict) -> None:
    steps = result.get("intermediate_steps", [])
    output = result.get("output", "")

    trace = langfuse.start_observation(
        name="sra-agent-run",
        input=query,
        metadata={
            "role": role,
            "iteration_count": len(steps),
            "agent_failed": not bool(output.strip()),
            "session_id": session_id or "no-session",
            "user_id": str(user_id),
        },
    )
    trace.update(output=output)

    for i, (action, observation) in enumerate(steps):
        obs_str = str(observation)
        status = "ERROR" if '"status": "ERROR"' in obs_str else \
                 "DENIED" if '"status": "DENIED"' in obs_str else "OK"
        tool_span = langfuse.start_observation(
            name=f"tool:{action.tool}",
            input=action.tool_input if isinstance(action.tool_input, dict)
                  else {"input": action.tool_input},
            metadata={"step": i + 1, "tool_status": status},
        )
        tool_span.update(output=obs_str)
        tool_span.end()

    trace.end()
    langfuse.flush()