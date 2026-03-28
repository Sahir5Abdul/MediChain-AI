import os
import sys
import json
import requests


def _base_url():
    # n8n typically runs on same machine as FastAPI (localhost:8000).
    # Allow override via env or CLI:
    # - env: FASTAPI_BASE_URL
    # - cli: python pipeline.py http://localhost:8000
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip().rstrip("/")
    return os.getenv("FASTAPI_BASE_URL", "http://localhost:8000").rstrip("/")


def _post(url, timeout_s=600):
    r = requests.post(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def main():
    base = _base_url()
    print("[pipeline] using FastAPI at {0}".format(base))

    # 1) Run deterministic pipeline (writes the JSON files FastAPI serves)
    run = _post("{0}/run-pipeline".format(base), timeout_s=900)
    print("[pipeline] /run-pipeline ok: run_id={0}".format(run.get("run_id")))

    # 2) Run LLM agents to produce agent1-4 outputs for dashboard reasoning panel
    agents = _post("{0}/agent/run-all".format(base), timeout_s=1800)
    print("[pipeline] /agent/run-all ok: status={0}".format(agents.get("status")))

    # 3) Print a compact summary n8n can log
    out = {
        "status": "ok",
        "run_id": run.get("run_id"),
        "pipeline": run,
        "agents": agents,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()