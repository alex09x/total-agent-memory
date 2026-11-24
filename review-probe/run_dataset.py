"""Load the synthetic dataset into a running TAM through MCP stdio and score the gold queries.

Usage: python3 -I run_dataset.py --cmd "tam" [--dataset DIR] [--restarts 2] [--out result.json]
Standard library only. Talks JSON-RPC to the server exactly as an IDE would.
"""
import argparse, json, os, shlex, statistics, subprocess, sys, time
from pathlib import Path

# A made-up, key-shaped string for the redaction check; assembled here so the fixture file holds no key-shaped literal.
FAKE_KEY = "sk_" + "test_" + "FAKE" * 5 + "1234"

class Mcp:
    def __init__(self, cmd, stderr_path):
        self.p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(stderr_path, "a"), text=True, bufsize=1)
        self.n = 0
        init = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "review-dataset", "version": "1"}})
        self.server = init["result"]["serverInfo"]
        self.rpc("notifications/initialized", notify=True)
        self.tools = [t["name"] for t in self.rpc("tools/list")["result"]["tools"]]
    def rpc(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None: msg["params"] = params
        if not notify: self.n += 1; msg["id"] = self.n
        self.p.stdin.write(json.dumps(msg) + "\n"); self.p.stdin.flush()
        if notify: return None
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("server closed the pipe")
            r = json.loads(line)
            if r.get("id") == self.n: return r
    def call(self, name, **args):
        args = {k: v for k, v in args.items() if v not in (None, [], "")}
        t = time.perf_counter(); r = self.rpc("tools/call", {"name": name, "arguments": args})
        ms = (time.perf_counter() - t) * 1000
        res = r.get("result") or {}
        txt = "".join(c.get("text", "") for c in res.get("content", []))
        try: body = json.loads(txt)
        except Exception: body = {"_text": txt}
        if r.get("error"): body = {"_error": r["error"]}
        return body, ms, bool(res.get("isError") or r.get("error"))
    def close(self):
        try: self.p.stdin.close(); self.p.terminate(); self.p.wait(timeout=20)
        except Exception: self.p.kill()

def items(body):
    res = body.get("results")
    if isinstance(res, dict): return [i for g in res.values() for i in g]
    if isinstance(res, list): return res
    return []

def pct(xs, p):
    xs = sorted(xs); return round(xs[min(len(xs) - 1, int(len(xs) * p))], 1) if xs else None

def run_queries(m, Q, ids):
    out, lat = [], []
    for q in Q:
        p = q["params"]
        args = dict(query=q["query"], project=q["project"], limit=q["k"], branch=p.get("branch"),
                    expand_context=p.get("expand_context"))
        if q["kind"] == "cache_poison":
            m.call("memory_recall", **args)
            m.call("memory_recall", **args, topics=["no-such-topic-review"])
        body, ms, err = m.call("memory_recall", **args)
        lat.append(ms)
        got = [i.get("id") for i in items(body)]
        exp_ids = [x.get("id") for x in body.get("expansion", [])]
        want = [[ids.get(a) for a in k.split("|")] for k in q["expect"]]
        hits = [any(w in got for w in alts) for alts in want]
        leaked = [k for k in q["absent"] if ids.get(k) in got or ids.get(k) in exp_ids]
        flags = {i.get("id"): [f for f in ("untrusted_instructions", "redacted", "truncated") if i.get(f)] for i in items(body)}
        ok = all(hits) and not leaked and not err
        if q["kind"] == "untrusted":
            ok = ok and any(flags.get(ids.get("injection")) or [])
        out.append({"id": q["id"], "kind": q["kind"], "ok": ok, "hits": hits, "leaked": leaked, "error": err,
                    "order": got, "tiers": body.get("tiers_used"), "ms": round(ms, 1),
                    "flags": {str(k): v for k, v in flags.items() if v}})
    return out, lat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True); ap.add_argument("--dataset", default=str(Path(__file__).parent))
    ap.add_argument("--restarts", type=int, default=2); ap.add_argument("--out", default="dataset-result.json")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    R = json.loads(Path(a.dataset, "records.json").read_text()); Q = json.loads(Path(a.dataset, "queries.json").read_text())
    err_log = os.path.abspath(a.out) + ".stderr"
    m = Mcp(a.cmd, err_log)
    ids, save_ms, save_notes = {}, [], {}
    for r in R:
        r["content"] = r["content"].replace("@@FAKE_KEY@@", FAKE_KEY)
        if r["update_of"]:
            body, ms, err = m.call("memory_update", id=ids[r["update_of"]], new_content=r["content"], reason="dataset update")
            ids[r["key"]] = body.get("new_id"); save_notes[r["key"]] = body
        else:
            body, ms, err = m.call("memory_save", type=r["type"], content=r["content"], project=r["project"],
                                   context=r["context"], tags=r["tags"], branch=r["branch"])
            ids[r["key"]] = body.get("id")
            if err or not body.get("saved", True) or body.get("deduplicated"): save_notes[r["key"]] = body
        save_ms.append(ms)
    stored_secret, _, _ = m.call("memory_get", ids=[ids["fake-secret"]])
    stats, _, _ = m.call("memory_stats")
    runs = []
    first, lat = run_queries(m, Q, ids); runs.append(first)
    m.close()
    for _ in range(max(0, a.restarts - 1)):
        m = Mcp(a.cmd, err_log); rr, l2 = run_queries(m, Q, ids); runs.append(rr); lat += l2; m.close()
    kinds = {}
    for r in first: kinds.setdefault(r["kind"], [0, 0]); kinds[r["kind"]][0] += r["ok"]; kinds[r["kind"]][1] += 1
    same = [all(run[i]["order"] == runs[0][i]["order"] for run in runs) for i in range(len(Q))]
    secret_txt = json.dumps(stored_secret)
    result = {"label": a.label, "server": m.server, "tools": len(m.tools),
              "saved": sum(1 for v in ids.values() if v), "records": len(R),
              "save_ms_p50": pct(save_ms, .5), "save_ms_p95": pct(save_ms, .95),
              "recall_ms_p50": pct(lat, .5), "recall_ms_p95": pct(lat, .95),
              "passed": sum(r["ok"] for r in first), "queries": len(Q),
              "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in kinds.items()},
              "identical_across_restarts": f"{sum(same)}/{len(same)}",
              "secret_stored_verbatim": FAKE_KEY in secret_txt,
              "failed": [r for r in first if not r["ok"]],
              "differs_across_restarts": [Q[i]["id"] for i, s in enumerate(same) if not s],
              "save_notes": save_notes, "stats": stats}
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    short = {k: result[k] for k in ("label", "server", "tools", "saved", "records", "save_ms_p50", "recall_ms_p50",
                                    "recall_ms_p95", "passed", "queries", "by_kind", "identical_across_restarts",
                                    "secret_stored_verbatim", "differs_across_restarts")}
    short["failed_ids"] = [(r["id"], r["hits"], r["leaked"]) for r in result["failed"]]
    print(json.dumps(short, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
