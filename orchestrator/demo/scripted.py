
import json
import re

from ..schemas import Complexity, ExecutionPlan, ReviewVerdict, Specialist, SpecialistReport, Subtask

ANALYSIS_TAIL = '''
import csv, io, statistics
rows = list(csv.DictReader(io.StringIO(raw.strip())))
for r in rows:
    r["qps"], r["p50_ms"], r["memory_gb"] = float(r["qps"]), float(r["p50_ms"]), float(r["memory_gb"])
print("rows: %d" % len(rows))
print("avg_qps: %.1f" % statistics.mean(r["qps"] for r in rows))
print("avg_p50_ms: %.1f" % statistics.mean(r["p50_ms"] for r in rows))
print("avg_memory_gb: %.2f" % statistics.mean(r["memory_gb"] for r in rows))
print("best_qps: " + max(rows, key=lambda r: r["qps"])["database"])
print("lowest_latency: " + min(rows, key=lambda r: r["p50_ms"])["database"])
print("lowest_memory: " + min(rows, key=lambda r: r["memory_gb"])["database"])
'''


def _sub(i, desc, spec, deps, out):
    return Subtask(id=i, description=desc, specialist=spec, depends_on=deps, required_inputs=[],
                   expected_output=out, complexity=Complexity.MEDIUM)


# ---------------- supervisor ----------------
def create_plan(task, previous_error=None, memory_context="", human_feedback=""):
    ctx = (memory_context or "").lower()
    learned = "citation" in ctx or "cite" in ctx            # memory says an earlier run was rejected for this
    d4 = "Write a recommendation on vector databases for the team using the research and the analysis."
    if learned:
        d4 += " Cite sources as [1], [2] and include the computed numbers."
    why = ("Memory: an earlier run of a similar task was sent back by the reviewer for missing citations, "
           "so the writing step asks for them up front." if learned else "No relevant memory: standard plan.")
    return ExecutionPlan(
        rationale=f"Search two topics in parallel, analyze the benchmark, then write. {why}", confidence=0.9,
        subtasks=[
            _sub("t1", "Search for what Chroma, Qdrant and pgvector are and who they suit",
                 Specialist.RESEARCH, [], "a sourced overview"),
            _sub("t2", "Search for benchmark figures (queries per second, latency, memory) for the three databases",
                 Specialist.RESEARCH, [], "the benchmark table"),
            _sub("t3", "Extract the benchmark table and compute averages and the best database per metric",
                 Specialist.DATA_ANALYSIS, ["t2"], "computed statistics"),
            _sub("t4", d4, Specialist.WRITING, ["t1", "t3"], "a short recommendation"),
        ])


# ---------------- specialists ----------------
def _csv_from(text: str) -> str:
    out, on = [], False
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("database,"):
            on, out = True, [ln]
        elif on and re.fullmatch(r"[\w-]+,[\d.]+,[\d.]+,[\d.]+", ln):
            out.append(ln)
        elif on:
            break
    return "\n".join(out) if len(out) > 1 else ""


def run_specialist(subtask, dep_outputs, feedback, registry):
    name, desc, fb = subtask.specialist.value, subtask.description, feedback or ""
    if "CRASH" in desc:
        raise RuntimeError("scripted failure")
    if "FLAKY" in desc and feedback is None:
        raise RuntimeError("scripted one-off failure")

    m = re.search(r"DELETE:\s*(\S+)", desc)
    if m:
        if "ALREADY executed" in fb:
            return SpecialistReport(answer=f"Deleted {m.group(1)} (approved by a human).", confidence=0.9), ""
        out = registry.invoke("delete_file", name, {"path": m.group(1)})     # raises ApprovalRequired
        return SpecialistReport(answer=f"Deleted {m.group(1)}.", confidence=0.9), \
            f"TOOL delete_file args={{'path': {m.group(1)!r}}} -> {out}"

    if name == "research":
        query = "benchmark queries per second latency" if "benchmark" in desc.lower() else "Chroma Qdrant pgvector overview"
        raw = registry.invoke("web_search", name, {"query": query, "max_results": 3})
        if raw.startswith("ERROR"):
            raise RuntimeError(raw)
        lines = [f"[{i}] {r['title']} ({r['url']}):\n{r['snippet']}"
                 for i, r in enumerate(json.loads(raw)["results"], 1)]
        return (SpecialistReport(answer="\n".join(lines), confidence=0.85),
                f"TOOL web_search args={{'query': {query!r}, 'max_results': 3}} -> {raw[:1500]}")

    if name == "data_analysis":
        csv_text = _csv_from("\n".join(dep_outputs.values()))
        if not csv_text:
            raise RuntimeError("no benchmark table found in the research output")
        code = "raw = " + repr(csv_text) + "\n" + ANALYSIS_TAIL
        raw = registry.invoke("run_python", name, {"code": code, "timeout_s": 20})
        if raw.startswith("ERROR"):
            raise RuntimeError(raw)
        stdout = json.loads(raw)["stdout"].strip()
        return SpecialistReport(answer=stdout, confidence=0.9), f"TOOL run_python args={{'code': '...'}} -> {raw[:1500]}"

    if name == "writing":
        n = dict(re.findall(r"^(\w+): (.+)$", dep_outputs.get("t3", ""), re.M))
        cite = any(w in (desc + " " + fb).lower() for w in ("cite", "citation"))
        c1, c2 = ("[1]", "[2]") if cite else ("", "")
        answer = (f"Recommendation: choose {n.get('best_qps', 'the fastest option')} for performance-critical work {c1}. "
                  f"Across the three databases the average throughput is {n.get('avg_qps', '?')} queries per second, "
                  f"with a median latency of {n.get('avg_p50_ms', '?')} ms and {n.get('avg_memory_gb', '?')} GB of memory {c2}. "
                  f"{n.get('lowest_latency', '?')} has the lowest latency and {n.get('lowest_memory', '?')} "
                  f"the lowest memory use {c2}. Chroma is the easiest to start with on a laptop {c1}, "
                  f"and pgvector suits teams that already run PostgreSQL {c1}.")
        return SpecialistReport(answer=answer, confidence=0.9), ""

    return SpecialistReport(answer=f"Done: {desc}", confidence=0.9), ""


# ---------------- reviewer ----------------
def review_output(subtask, report, evidence=""):
    name, ans = subtask.specialist.value, report.answer

    def no(msg):
        return ReviewVerdict(approved=False, feedback=msg, confidence=0.9)

    if name == "research" and "TOOL web_search" not in evidence:
        return no("Use the web_search tool and base the answer on its results.")
    if name == "data_analysis" and ("TOOL run_python" not in evidence or "avg_qps" not in evidence):
        return no("Run the analysis with run_python and report its real output.")
    if name == "writing" and len(re.findall(r"\[\d+\]", ans)) < 2:
        return no("Add source citations like [1] and [2] for the factual claims.")
    return ReviewVerdict(approved=True, feedback="ok", confidence=0.9)


# ---------------- final answer ----------------
def synthesize(task, outputs):
    final = outputs.get("t4") or "\n\n".join(outputs.values())
    if outputs.get("t3"):
        final += "\n\nBenchmark analysis (computed by run_python):\n" + outputs["t3"]
    return final + "\n\nNote: the sources and figures are fictional demo data."