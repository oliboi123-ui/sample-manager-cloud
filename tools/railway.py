"""Railway checks from a PC: the project's services, their variables (secret values masked) and recent deployments, and
a deployment's log. Needs a Railway *project token* (project Settings -> Tokens) in ~/.railway-token; it is never printed.
    python tools/railway.py look
    python tools/railway.py logs <deployment id> [lines]
    python tools/railway.py deploy [commit]     the newest commit by default; waits for it (pushing alone doesn't deploy)
    python tools/railway.py gql '<query>' ['<variables json>']     (careful: prints what it returns, secrets included)
"""
import json
import os
import re
import sys
import urllib.request

URL = "https://backboard.railway.com/graphql/v2"
TOKEN = open(os.path.expanduser("~/.railway-token"), encoding="ascii").read().strip()


def gql(query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json", "Project-Access-Token": TOKEN,
                                             "User-Agent": "sample-manager-checks"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    if out.get("errors"):
        raise SystemExit("errors: " + json.dumps(out["errors"])[:800])
    return out["data"]


def mask(name, raw, rendered):
    if raw is None:
        return "(not set)"
    if "${{" in raw:
        shown = raw
    elif raw == "":
        shown = "(empty)"
    elif name in ("AUTO_PUBLISH", "TRAIN_HOUR_UTC", "MAX_PER_INSTALL", "PORT") or name.startswith("RAILWAY_"):
        shown = raw
    else:
        shown = f"(secret, {len(raw)} chars)"
    if rendered is not None and rendered != raw:
        m = re.match(r"^([a-z0-9+]+://)?", rendered)
        shown += f"  -> renders as {m.group(1) + '...' if m and m.group(1) else ('(empty)' if rendered == '' else f'{len(rendered)} chars')}"
    return shown


def look():
    ids = gql("query { projectToken { projectId environmentId } }")["projectToken"]
    p, e = ids["projectId"], ids["environmentId"]
    proj = gql("query($id: String!) { project(id: $id) { name services { edges { node { id name } } } "
               "environments { edges { node { id name } } } } }", {"id": p})["project"]
    envs = {x["node"]["id"]: x["node"]["name"] for x in proj["environments"]["edges"]}
    print(f"project {proj['name']}, environment {envs.get(e, e)}")
    for s in [x["node"] for x in proj["services"]["edges"]]:
        print(f"\nservice {s['name']!r} ({s['id']})")
        q = "query($p: String!, $e: String!, $s: String!, $u: Boolean) { variables(projectId: $p, environmentId: $e, serviceId: $s, unrendered: $u) }"
        raw = gql(q, {"p": p, "e": e, "s": s["id"], "u": True})["variables"]
        ren = gql(q, {"p": p, "e": e, "s": s["id"], "u": False})["variables"]
        for k in sorted(raw):
            if k.startswith("RAILWAY_"):
                continue
            print(f"  {k} = {mask(k, raw[k], ren.get(k))}")
        d = gql("query($p: String!, $e: String!, $s: String!) { deployments(first: 4, input: {projectId: $p, environmentId: $e, serviceId: $s}) "
                "{ edges { node { id status createdAt } } } }", {"p": p, "e": e, "s": s["id"]})["deployments"]["edges"]
        for x in d:
            n = x["node"]
            print(f"  deploy {n['createdAt'][:19]}  {n['status']:<10} {n['id']}")


def deploy(sha=None):
    """Deploys a commit (the newest of this repo by default) and waits for it (a push alone doesn't: no GitHub trigger)."""
    import subprocess
    import time
    ids = gql("query { projectToken { projectId environmentId } }")["projectToken"]
    proj = gql("query($id: String!) { project(id: $id) { services { edges { node { id name } } } } }", {"id": ids["projectId"]})["project"]
    service = next(x["node"]["id"] for x in proj["services"]["edges"] if x["node"]["name"] == "sample-manager-cloud")
    sha = sha or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    dep = gql("mutation($e: String!, $s: String!, $c: String) { serviceInstanceDeployV2(environmentId: $e, serviceId: $s, commitSha: $c) }",
              {"e": ids["environmentId"], "s": service, "c": sha})["serviceInstanceDeployV2"]
    print(f"deploying {sha[:7]}: {dep}", flush=True)
    for _ in range(90):
        status = gql("query($d: String!) { deployment(id: $d) { status } }", {"d": dep})["deployment"]["status"]
        if status in ("SUCCESS", "FAILED", "CRASHED", "REMOVED", "SKIPPED"):
            break
        time.sleep(10)
    print(f"deploy {dep}: {status}")
    if status != "SUCCESS":
        raise SystemExit(f"see: python tools/railway.py logs {dep}")


def logs(dep, n=60):
    out = gql("query($d: String!, $n: Int) { deploymentLogs(deploymentId: $d, limit: $n) { timestamp message } }", {"d": dep, "n": int(n)})
    for line in out["deploymentLogs"]:
        print(line["timestamp"][11:19], line["message"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "look"
    if cmd == "look":
        look()
    elif cmd == "logs":
        logs(*sys.argv[2:])
    elif cmd == "deploy":
        deploy(*sys.argv[2:])
    elif cmd == "gql":
        print(json.dumps(gql(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None), indent=1))
