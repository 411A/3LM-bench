#!/usr/bin/env python3
"""Smoke test: prove every backend code path works before committing to a long run."""
import json
import sys

from backend import Backend
from scorers import (check_devops, extract_code, lint_python, lint_rust, normalize_fa,
                     run_python, run_rust, score_netdiag)

kind = sys.argv[1] if len(sys.argv) > 1 else "ollama"
name = sys.argv[2] if len(sys.argv) > 2 else "nemotron-3-nano:4b"
b = Backend(kind, name, port=11434 if kind == "ollama" else 8090,
            api_key=None if kind == "ollama" else "local-llama")
print(f"backend={kind} model={name} alive={b.alive()}")

# 1. plain generation
r = b.generate([{"role": "user", "content": "Reply with exactly: OK"}],
               max_tokens=8, temperature=0.0)
print(f"1. generate      : {r['content']!r}  gen_tps={r['gen_tps']} tokens={r['eval_tokens']}")

# 2. MC via single-token logprobs (6-tuple: backend also returns letter_mass)
_mc = b.mc_logprobs(
    "Which is a prime number?", ["4", "6", "7", "9"])
idx, letter, dist, meta, mode = _mc[:5]
print(f"2. mc_logprobs   : chose={letter} idx={idx} gold=2 mode={mode} "
      f"dist={{ {', '.join(f'{k}:{v:.2f}' for k, v in sorted(dist.items()))} }} tokens={meta['eval_tokens']}")

# 3. code generation + execution
code = extract_code(b.generate([{"role": "user", "content":
    "Write a Python function `add(a, b)` returning a + b. Reply with only a ```python block."}],
    max_tokens=120)["content"], "python")
ex = run_python(code, "assert add(2, 3) == 5")
print(f"3. code exec     : passed={ex['passed']} lint={lint_python(code)['n_diagnostics']} diags")

# 4. rust
rs = extract_code(b.generate([{"role": "user", "content":
    "Write only this Rust function, no fn main, no #[test]: `fn double(x: i32) -> i32`"}],
    max_tokens=140)["content"], "rust")
rr = run_rust(rs, ["assert_eq!(double(3), 6);"], "double")
print(f"4. rust exec     : passed={rr['passed']} reason={rr['error']} wrote_main={rr.get('wrote_main')}")

# 5. tool call parse
TOOLS = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}}]
tr = b.generate([{"role": "user", "content": "What is the weather in Tehran? Use the tool."}],
                max_tokens=160, tools=TOOLS)
tcs = tr["tool_calls"]
print(f"5. tool call     : {json.dumps(tcs)[:150]}")

print("summary:", json.dumps(b.summary()))