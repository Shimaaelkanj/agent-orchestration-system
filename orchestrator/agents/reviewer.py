
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm, structured
from ..schemas import ReviewVerdict, SpecialistReport, Subtask

REVIEW_PROMPT = """You are a reviewer. Decide whether the specialist's output accomplishes the TASK and
matches the EXPECTED OUTPUT. Nothing else.

Rules:
- APPROVE if the output does what the task literally asks, even if it is simple or imperfect.
- REJECT only if something the task explicitly requires is missing, wrong, or clearly fabricated.
- TOOL EVIDENCE is the ground truth. If the output reports results that contradict or are not
  supported by the tool evidence, REJECT and name the discrepancy.
- NEVER reject for style: no demands for docstrings, type hints, renaming, extra features,
  input validation, or best practices that the task did not ask for.
- NEVER write the solution yourself. If rejecting, name the ONE problem in one sentence.
- Keep feedback under 40 words.
- Expected output is a DESCRIPTION of the desired result, not exact words or headings that must appear.
  Judge the meaning, not the labels."""


def review_output(subtask: Subtask, report: SpecialistReport, evidence: str = "") -> ReviewVerdict:
    llm = structured(get_llm("reviewer"), "reviewer", ReviewVerdict)
    msg = (f"Task: {subtask.description}\nExpected output: {subtask.expected_output}\n\n"
           f"Specialist output:\n{report.answer}\n\n"
           f"TOOL EVIDENCE (ground truth):\n{evidence or 'No tools were called.'}\n\n"
           f"Specialist confidence: {report.confidence}")
    v = llm.invoke([SystemMessage(REVIEW_PROMPT), HumanMessage(msg)])
    v.confidence = max(0.0, min(1.0, v.confidence))
    return v