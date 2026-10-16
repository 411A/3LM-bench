#!/usr/bin/env python3
"""
DevOps scorer control: the echo must fail, real artifacts must pass.

This is a permanent fixture. It exists because the scorer regressed in BOTH
directions at different times:

  * substring-only matching let a one-line keyword echo score 1.0 on 10/10, so
    the suite rewarded a model that wrote no file at all;
  * over-correcting to "each requirement must anchor a line" then scored genuine
    Dockerfiles and compose files at 0.5-0.88, because YAML list items, inline
    FROM ... AS build, and values under a key never anchor a line.

A scorer for artifacts has to satisfy both halves at once. This file holds the
adversarial inputs, not the happy path.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scorers as S  # noqa: E402

TASKS = json.load(open(os.path.join(HERE, "tasks", "devops.json")))

# A genuinely correct artifact per task, written by hand against each prompt.
REAL = {
"devops/0": "FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app\n"
            "RUN pip install --no-cache-dir fastapi uvicorn\nEXPOSE 8000\n"
            "USER appuser\n"
            "HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1\n"
            'CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]',
"devops/1": "FROM golang:1.22 AS build\nWORKDIR /src\nCOPY . .\nRUN go build -o /out/app\n"
            "FROM alpine:3.19\nCOPY --from=build /out/app /usr/local/bin/app\n"
            'ENTRYPOINT ["/usr/local/bin/app"]',
"devops/2": 'version: "3.8"\nservices:\n  api:\n    image: nginx:1.25\n'
            "    ports:\n      - \"80:80\"\n  db:\n    image: postgres:16\n"
            "    volumes:\n      - pgdata:/var/lib/postgresql/data\n"
            "volumes:\n  pgdata:",
"devops/3": 'version: "3.8"\nservices:\n  web:\n    image: nginx\n'
            "    depends_on:\n      db:\n        condition: service_healthy\n"
            "    healthcheck:\n      test: [\"CMD\",\"curl\",\"-f\",\"http://localhost\"]\n"
            "  db:\n    image: postgres\n",
"devops/4": "apiVersion: apps/v1\nkind: Deployment\nspec:\n  template:\n    spec:\n"
            "      securityContext:\n        runAsNonRoot: true\n"
            "      containers:\n      - name: app\n        image: app:1\n"
            "        readinessProbe:\n          httpGet:\n            path: /health\n"
            "            port: 8080\n---\napiVersion: v1\nkind: Service\nspec:\n"
            "  ports:\n  - port: 80\n",
"devops/5": "apiVersion: apps/v1\nkind: Deployment\nspec:\n  selector:\n"
            "    matchLabels:\n      app: web\n  template:\n    metadata:\n"
            "      labels:\n        app: web\n    spec:\n      containers:\n"
            "      - name: web\n        image: nginx:1.25\n",
"devops/6": "name: ci\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-go@v5\n",
"devops/7": "name: release\non:\n  push:\n    tags: ['v*']\njobs:\n  ship:\n"
            "    runs-on: ubuntu-latest\n    steps:\n"
            "      - if: github.ref == 'refs/heads/main'\n        run: ./deploy.sh\n"
            "        env:\n          TOKEN: ${{ secrets.DEPLOY_TOKEN }}\n",
"devops/8": "[Unit]\nDescription=api\nAfter=network.target\n[Service]\n"
            "ExecStart=/usr/local/bin/api\nRestart=always\nUser=api\n"
            "[Install]\nWantedBy=multi-user.target",
"devops/9": "server {\n  listen 80;\n  location /api {\n"
            "    proxy_pass http://127.0.0.1:8000;\n    proxy_set_header Host $host;\n"
            "  }\n  location / {\n    root /var/www;\n  }\n}\n",
}

fails = []


def note(ok, msg):
    print(f"  {'OK ' if ok else 'BUG'} {msg}")
    if not ok:
        fails.append(msg)


print("=== the keyword echo must NOT score 1.0 (it is not a file) ===")
perfect = 0
for t in TASKS:
    echo = " ".join(t["checks"].get("must_contain", []))
    r = S.check_devops(echo, t["checks"])
    perfect += (r["score"] == 1.0)
note(perfect == 0, f"echo scores 1.0 on {perfect}/{len(TASKS)} tasks (must be 0)")

print()
print("=== a 100%-comment keyword salad is not an artifact ===")
salad_ok = True
for t in TASKS:
    salad = "\n".join(f"# {c}  (a comment, not an artifact line)"
                      for c in t["checks"].get("must_contain", []))
    r = S.check_devops(salad, t["checks"])
    if r["score"] == 1.0:
        salad_ok = False
        print(f"      {t['id']}: comment salad scored 1.0")
note(salad_ok, "comment-only salad rejected on all tasks")

print()
print("=== echo at EVERY length 1..N must be rejected ===")
for ln in range(1, 6):
    perfect = 0
    for t in TASKS:
        need = t["checks"].get("must_contain", [])
        if len(need) < ln:
            continue
        echo = "\n".join(need[:ln])
        r = S.check_devops(echo, t["checks"])
        perfect += (r["score"] == 1.0)
    note(perfect == 0, f"{ln}-line echo scores 1.0 on {perfect} tasks (must be 0)")

print()
print("=== FENCED and UNFENCED must score identically ===")
# This gap let a regression through: _strip_preamble ran a trailing
# "cut back to the last code-looking line" heuristic on fenced non-Python
# content and deleted most of a correct nginx config, so the FENCED version of
# a correct artifact scored 0.0 while the byte-identical unfenced one scored
# 1.0. Models fence constantly, so that penalised correct answers in the exact
# form they are written.
mismatch = []
for t in TASKS:
    art = REAL.get(t["id"])
    if not art:
        continue
    for tag in ("```\n", "```" + (t["checks"].get("lang") or "text") + "\n"):
        pass
    unf = S.check_devops(art, t["checks"])["score"]
    for lang in ("nginx", "yaml", "dockerfile", ""):
        f = "```" + lang + "\n" + art + "\n```"
        fenc = S.check_devops(f, t["checks"])["score"]
        if fenc != unf:
            mismatch.append((t["id"], lang or "(none)", unf, fenc))
note(not mismatch, f"fenced == unfenced on all {len(REAL)} artifacts"
     + (f" (mismatches: {mismatch})" if mismatch else ""))

print()
print("=== genuine artifacts MUST still score 1.0 (no over-correction) ===")
ok = 0
for t in TASKS:
    art = REAL.get(t["id"])
    if not art:
        continue
    r = S.check_devops(art, t["checks"])
    ok += (r["score"] == 1.0)
    if r["score"] != 1.0:
        print(f"      {t['id']}: score={r['score']} missed={r['missed']} "
              f"lines={r.get('lines')}")
note(ok == len(REAL), f"real artifacts score 1.0 on {ok}/{len(REAL)} (must be all)")

print()
print("=== a banned string must still be penalised ===")
t = next(x for x in TASKS if x["id"] == "devops/0")
bad = REAL["devops/0"].replace("USER appuser", "USER root")
r = S.check_devops(bad, t["checks"])
note(r["score"] < 1.0 and "user root" in r["violations"],
     f"USER root penalised: score={r['score']} violations={r['violations']}")

print()
print("=" * 68)
if fails:
    print(f"{len(fails)} FAILURES")
    for f in fails:
        print(f"   - {f}")
else:
    print("DEVOPS SCORER OK: rejects keyword echoes, accepts real artifacts.")
sys.exit(1 if fails else 0)