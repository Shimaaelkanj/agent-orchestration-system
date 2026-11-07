from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..llm import get_llm, structured
from ..schemas import SpecialistReport, Subtask
from ..tools import ApprovalRequired, ToolRegistry

MAX_STEPS = 5

ROLE_PROMPTS = {
    "research": "You are a research specialist. Find accurate, sourced information using your tools.",
    "data_analysis": "You are a data analyst. Query data and compute results; show numbers, not guesses.",
    "writing": "You are a writing specialist. Your final answer must BE the requested text itself, "
               "not a description of it and not a file path. Use only the earlier subtask outputs given "
               "to you, quoting their concrete results. Do not invent results.",
    "code_execution": "You are a coding specialist. Write code, run it, and report verified results.",
}


def _as_text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def run_specialist(subtask: Subtask, dep_outputs: dict[str, str], feedback: str | None,
                   registry: ToolRegistry) -> tuple[SpecialistReport, str]:
    name = subtask.specialist.value
    llm = get_llm("specialist")
    tools = registry.tools_for(name)
    by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    prompt = f"Task: {subtask.description}\nExpected output: {subtask.expected_output}\n"
    if dep_outputs:
        prompt += "\nOutputs from earlier subtasks:\n" + "\n".join(f"[{k}] {v}" for k, v in dep_outputs.items())
    if feedback:
        prompt += f"\n\nNOTES FROM PREVIOUS ATTEMPTS, THE REVIEWER OR A HUMAN (follow them):\n{feedback}"

    rules = "Copy any results EXACTLY from the provided inputs or tool output. Never invent results."
    if name in ("code_execution", "data_analysis"):
        rules += (" Code you run must print() every result you need to report. If a tool returns an error "
                  "or empty output, fix your code and run it again; if it still fails, say so plainly.")
    rules += (" If the task genuinely requires a sensitive tool (deleting data, sending messages), call it "
              "directly: the system will ask a human for approval first.")
    prompt += "\n\nRULES: " + rules

    messages = [SystemMessage(ROLE_PROMPTS[name]), HumanMessage(prompt)]
    evidence: list[str] = []
    for _ in range(MAX_STEPS):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            break
        for call in ai.tool_calls:
            tool = by_name.get(call["name"])
            try:
                out = tool.invoke(call["args"]) if tool else f"ERROR: unknown tool {call['name']}"
            except ApprovalRequired as a:
                a.reasoning = _as_text(ai.content)[:500]     # why the agent wanted this action
                raise
            evidence.append(f"TOOL {call['name']} args={call['args']} -> {str(out)[:1500]}")
            messages.append(ToolMessage(content=str(out), tool_call_id=call["id"]))

    messages.append(HumanMessage("Give your final answer. Copy any results EXACTLY from the tool outputs "
                                 "above; never invent output. Then give an honest confidence score between 0 and 1."))
    report = structured(llm, "specialist", SpecialistReport).invoke(messages)
    report.confidence = max(0.0, min(1.0, report.confidence))
    return report, "\n".join(evidence)
