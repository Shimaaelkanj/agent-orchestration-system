
"""Offline 'web': fictional pages used by the demo. The numbers are made up for illustration."""

CORPUS = [
    {"kind": "overview", "title": "Chroma overview", "url": "https://demo.invalid/chroma",
     "content": "Chroma is an open-source embedding database that is easy to run locally and is Python-first. "
                "It suits prototypes and small teams (fictional demo data)."},
    {"kind": "overview", "title": "Qdrant overview", "url": "https://demo.invalid/qdrant",
     "content": "Qdrant is a vector search engine written in Rust with rich filtering. "
                "It suits teams that need high throughput (fictional demo data)."},
    {"kind": "overview", "title": "pgvector overview", "url": "https://demo.invalid/pgvector",
     "content": "pgvector is a PostgreSQL extension that adds vector search to a relational database, "
                "with transactions and SQL joins (fictional demo data)."},
    {"kind": "benchmark", "title": "Benchmark table (fictional)", "url": "https://demo.invalid/benchmark",
     "content": "database,qps,p50_ms,memory_gb\nchroma,1200,8,6.1\nqdrant,3400,4,5.2\npgvector,900,11,7.4\n"},
    {"kind": "benchmark", "title": "Benchmark method (fictional)", "url": "https://demo.invalid/method",
     "content": "Single node, one million vectors, 768 dimensions, queries per second and median latency "
                "(fictional demo data)."},
]


def search(query: str, n: int = 3) -> list[dict]:
    kind = "benchmark" if "benchmark" in query.lower() else "overview"
    pages = [p for p in CORPUS if p["kind"] == kind]
    return [{"title": p["title"], "url": p["url"], "snippet": p["content"]} for p in pages[:n]]