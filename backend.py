#!/usr/bin/env python3
"""
Uniform client over the two local runtimes.

Everything downstream (task runners, scorers) talks to Backend and never to
urllib directly, so adding a third runtime is one class.

Timing detail that matters: raw `eval_count / eval_duration` from llama.cpp is
the server's own counter and is the honest number. For Ollama we derive the same
thing from eval_count and total duration. Both are reported, plus a wall-clock
measured client-side figure, because on a 2-core box the first request pays a
page-cache miss that inflates it and subsequent ones do not.
"""
import json
import re
import time
import urllib.error
import urllib.request


class BackendError(RuntimeError):
    pass


class Backend:
    def __init__(self, kind, name, host=None, port=None, api_key=None, timeout=900,
                 num_ctx=4096, n_gpu_layers=0, n_threads=None, model_path=None):
        self.kind = kind                      # "ollama" | "llamacpp"
        self.name = name                      # ollama tag, or a label for llama.cpp
        self.host = host or ("127.0.0.1" )
        self.port = port or (11434 if kind == "ollama" else 8090)
        self.api_key = api_key
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.model_path = model_path
        self.n_calls = 0
        self.prompt_tokens = 0
        self.eval_tokens = 0
        self.wall = 0.0

    # ------------------------------------------------------------------ utils
    def model_digest(self):
        """Identify the exact bytes being benchmarked (eval-card requirement).

        Returns the registry digest for an ollama tag, or the sha256 of the
        GGUF file for a llama.cpp server. A bare tag ("qwen3.5:4b-q4_K_M") is
        not an identity: tags are mutable, digests are not.
        """
        try:
            if self.kind == "ollama":
                with urllib.request.urlopen(
                        f"http://{self.host}:{self.port}/api/tags",
                        timeout=10) as r:
                    for m in json.loads(r.read()).get("models", []):
                        if m.get("name") == self.name:
                            return {"digest": m.get("digest"),
                                    "size_bytes": m.get("size")}
                return {"digest": None, "note": "tag not in /api/tags"}
            if self.model_path:
                import hashlib
                h = hashlib.sha256()
                with open(self.model_path, "rb") as f:
                    for blk in iter(lambda: f.read(1 << 20), b""):
                        h.update(blk)
                return {"digest": "sha256:" + h.hexdigest(),
                        "size_bytes": __import__("os").path.getsize(self.model_path)}
            return {"digest": None, "note": "no model_path for llamacpp"}
        except Exception as e:  # noqa: BLE001
            return {"digest": None, "note": f"{type(e).__name__}: {e}"[:120]}

    def evict(self):
        """Unload the model from the server NOW (not after keep-alive expiry).

        The runner process grows over a long run (measured: 2.8GB resident at
        load, 6.8GB when the OOM-killer took it mid-sweep). Evicting between
        tasks bounds that growth to one task's worth. Reload costs ~10s of
        mmap; losing a rep to OOM costs 50 items. Returns True if the server
        reports no loaded models afterwards (ollama only), "persistent" for
        llamacpp (the server owns the model; run_llama.sh restarts it per
        model, nothing per task).
        """
        if self.kind == "llamacpp":
            return "persistent"
        try:
            if self.kind == "ollama":
                import shutil
                import subprocess
                if shutil.which("ollama"):
                    subprocess.run(["ollama", "stop", self.name],
                                   capture_output=True, timeout=60)
                with urllib.request.urlopen(
                        f"http://{self.host}:{self.port}/api/ps",
                        timeout=10) as r:
                    return len(json.loads(r.read()).get("models", [])) == 0
        except Exception:  # noqa: BLE001
            pass
        return None

    def server_versions(self):
        """Backend + toolchain versions for the eval-card. Best-effort strings."""
        out = {}
        try:
            if self.kind == "ollama":
                with urllib.request.urlopen(
                        f"http://{self.host}:{self.port}/api/version",
                        timeout=10) as r:
                    out["ollama"] = json.loads(r.read()).get("version")
        except Exception:  # noqa: BLE001
            out["ollama"] = None
        import shutil
        import subprocess
        for tool, argv in (("ruff", ["ruff", "--version"]),
                           ("rustc", ["rustc", "--version"]),
                           ("g++", ["g++", "--version"]),
                           ("python", ["python3", "--version"])):
            try:
                if shutil.which(argv[0]):
                    p = subprocess.run(argv, capture_output=True, text=True,
                                       timeout=10)
                    out[tool] = (p.stdout or p.stderr).strip().splitlines()[0][:80]
                else:
                    out[tool] = None
            except Exception:  # noqa: BLE001
                out[tool] = None
        return out
    def _post(self, path, payload, extra_headers=None):
        url = f"http://{self.host}:{self.port}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        for k, v in (extra_headers or {}).items():
            req.add_header(k, v)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise BackendError(f"HTTP {e.code} from {url}: {detail}") from None
        except Exception as e:  # noqa: BLE001
            raise BackendError(f"{type(e).__name__} calling {url}: {e}") from None
        return json.loads(body), time.monotonic() - t0

    def alive(self):
        try:
            req = urllib.request.Request(f"http://{self.host}:{self.port}/health")
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:  # noqa: BLE001
            # ollama has no /health; fall back to a tags/models probe
            try:
                p = "/api/tags" if self.kind == "ollama" else "/v1/models"
                with urllib.request.urlopen(
                        f"http://{self.host}:{self.port}{p}", timeout=10) as r:
                    json.loads(r.read())
                return True
            except Exception:  # noqa: BLE001
                return False

    # ------------------------------------------------------------- generation
    def generate(self, messages, *, max_tokens=512, temperature=0.7, top_p=0.95,
                 stop=None, tools=None, logprobs=False, top_logprobs=20,
                 seed=None, think=False):
        """Returns dict(content, tool_calls, logprobs_first, tokens, tps, ttft, wall)."""
        if self.kind == "ollama":
            return self._ollama(messages, max_tokens, temperature, top_p, stop,
                                tools, logprobs, top_logprobs, seed, think)
        return self._llamacpp(messages, max_tokens, temperature, top_p, stop,
                              tools, logprobs, top_logprobs, seed)

    def _ollama(self, messages, max_tokens, temperature, top_p, stop, tools,
                logprobs, top_logprobs, seed, think):
        payload = {
            "model": self.name,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens,
                        "temperature": temperature, "top_p": top_p,
                        "num_gpu_layers": self.n_gpu_layers},
        }
        if stop:
            payload["options"]["stop"] = stop
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["options"]["seed"] = seed
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        j, wall = self._post("/api/chat", payload)
        msg = j.get("message") or {}
        pc = int(j.get("prompt_eval_count") or 0)
        ec = int(j.get("eval_count") or 0)
        ed = float(j.get("eval_duration") or 0) / 1e9
        pd = float(j.get("prompt_eval_duration") or 0) / 1e9
        ld = float(j.get("load_duration") or 0) / 1e9
        cached = int(j.get("prompt_eval_cached_count") or 0)
        self._acc(pc, ec, wall)
        lp = j.get("logprobs")
        first = None
        if isinstance(lp, list) and lp:
            first = lp[0]
        elif isinstance(lp, dict) and lp.get("content"):
            first = lp["content"][0]
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "thinking": msg.get("thinking") or "",
            "first_token_top": (first or {}).get("top_logprobs") or [],
            "prompt_tokens": pc, "eval_tokens": ec,
            "prompt_tps": round(pc / pd, 2) if pd else None,
            "gen_tps": round(ec / ed, 2) if ed else None,
            # pp/tg/load split: a single blended tok/s hides WHERE time goes.
            # load_duration contaminates the first request after a model swap
            # (cold mmap/page cache); prompt_eval_cached_count marks trials
            # whose prompt prefix was served from KV cache, which inflates pp
            # and must not be compared against cold trials.
            "load_s": round(ld, 3),
            "prompt_cached_tokens": cached,
            "ttft_s": round(pd, 3), "wall_s": round(wall, 3),
            "done_reason": j.get("done_reason"),
        }

    def _llamacpp(self, messages, max_tokens, temperature, top_p, stop, tools,
                  logprobs, top_logprobs, seed):
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "cache_prompt": True,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["seed"] = seed
        if logprobs:
            # /completion exposes raw top-k at each position; the chat route maps
            # it into OpenAI-shaped logprobs.
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        hdrs = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        j, wall = self._post("/v1/chat/completions", payload, hdrs)
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        usage = j.get("usage") or {}
        pc = int(usage.get("prompt_tokens") or 0)
        ec = int(usage.get("completion_tokens") or 0)
        tps = (j.get("timings") or {}).get("predicted_per_second")
        # llama.cpp reports the true rate in `timings`; fall back to usage math.
        gen_tps = round(tps, 2) if tps else (round(ec / (wall - 0.1), 2) if ec and wall > 0.1 else None)
        ttft = (j.get("timings") or {}).get("prompt_ms")
        self._acc(pc, ec, wall)
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "thinking": (msg.get("reasoning_content") or msg.get("reasoning") or ""),
            "first_token_top": _first_top_llamacpp(ch),
            "prompt_tokens": pc, "eval_tokens": ec,
            "prompt_tps": None, "gen_tps": gen_tps,
            "ttft_s": round(ttft / 1000, 3) if ttft else None,
            "wall_s": round(wall, 3),
            "done_reason": ch.get("finish_reason"),
        }

    def _acc(self, pc, ec, wall):
        self.n_calls += 1
        self.prompt_tokens += pc
        self.eval_tokens += ec
        self.wall += wall

    # --------------------------------------------------------------- MC by logprob
    def mc_logprobs(self, question, choices, system=None, prefill=""):
        """Answer a multiple-choice item with ONE forward pass.

        The prompt ends right where the answer letter goes, so we ask for a
        single token and read the model's own top-k distribution over the option
        letters. At 3-8 tok/s this turns a ~40-token generation into ~1 token,
        which is the difference between a full suite finishing in hours and in
        minutes. Returns (chosen_index, letter, distribution, meta).

        Confidence is the ABSOLUTE probability mass on the letter set - not the
        renormalised value. Renormalising inside the letters makes confidence
        structurally incapable of falling below 1/len(letters), so a model
        committing 5% of its mass to any answer still reported confidence 1.00.
        Run.py gates on this so a model with no real preference is recorded as
        low-information rather than silently scored at 50% by argmax.

        Returns (chosen_index, letter, normalised_distribution, meta, mode,
        letter_mass).
        """
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        opts = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
        prompt = (f"{question}\n\n{opts}\n\n"
                  f"Answer with a single letter ({'/'.join(letters)}) and nothing else.\nAnswer:")
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        if prefill:
            msgs[-1]["content"] += " " + prefill
        r = self.generate(msgs, max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20)
        top = r["first_token_top"] or []
        dist, n_bad_entries = {}, 0
        for d in top:
            if not isinstance(d, dict):
                n_bad_entries += 1
                continue
            # Raw token first, then normalised: leading "(" must not survive,
            # because "(A" is a standard answer form that the previous strip set
            # rejected, dropping the model's real choice and forcing the fallback.
            raw = d.get("token")
            if not isinstance(raw, str) or not raw:
                n_bad_entries += 1
                continue
            t = raw.strip().strip("*():.,\"").strip().upper()
            if len(t) != 1 or t not in letters:
                continue
            # Dispatch on the KEY PRESENT, never on the sign. logprob==0.0 and
            # probability==0.0 mean opposite things, and the old `lp > 0` test
            # applied the logprob branch to both, turning an impossible token
            # into probability 1.0. (Ollama's TokenLogprob has no probability
            # field, so that key never appears from this backend - but a
            # backend that does supply one would silently invert the answer.)
            if d.get("logprob") is not None:
                lp = d.get("logprob")
                if isinstance(lp, str) or not isinstance(lp, (int, float)):
                    n_bad_entries += 1
                    continue
                v = 2.718281828459045 ** lp
            elif d.get("probability") is not None:
                v = d.get("probability")
                if isinstance(v, str) or not isinstance(v, (int, float)):
                    n_bad_entries += 1
                    continue
            else:
                n_bad_entries += 1
                continue
            if v != v or v in (float("inf"), float("-inf")) or v < 0:
                n_bad_entries += 1
                continue
            # Accumulate. A top-k over token STRINGS routinely contains "A",
            # "A.", "**A" and "A)" as separate entries; assignment let whichever
            # happened to be serialised last win, so the argmax depended on JSON
            # key order rather than on the model.
            dist[t] = dist.get(t, 0.0) + v

        tot = sum(dist.values())
        if not dist or not tot > 0:
            # No letter mass at all (all-zero, or every entry malformed) - fall
            # back to the generated letter. Guard against an empty dict reaching
            # max(), which used to raise ValueError and abort the whole run.
            gen = (r["content"] or "").strip().upper()
            gen = gen.strip("*():.,\"").strip()
            m = re.search(r"[A-Z]", gen)
            L = m.group(0) if (m and m.group(0) in letters) else None
            idx = letters.index(L) if L else -1
            onehot = {letters[i]: (1.0 if i == idx else 0.0) for i in range(len(letters))}
            return (idx, (L or "?"), onehot, r, "generated")

        norm = {k: v / tot for k, v in dist.items()}
        if len(norm) == 1:
            best = next(iter(norm))
        else:
            # Index-stable tiebreak: heavily quantised models clamp many logits to
            # the same value, and dict insertion order made those runs depend on
            # backend serialisation order.
            best = min(norm, key=lambda k: (-norm[k], letters.index(k)))
        # Absolute letter mass, NOT the renormalised value. Renormalising inside
        # the letter set makes confidence structurally incapable of dropping
        # below 1/len(letters), so a model committing 5% of its mass to any
        # answer still reported confidence 1.00.
        letter_mass = min(1.0, tot)
        return (letters.index(best), best, norm, r, "logprob", letter_mass)

    def summary(self):
        return {
            "calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "wall_s": round(self.wall, 1),
            "mean_gen_tps": round(self.eval_tokens / self.wall, 2) if self.wall else None,
        }


def _first_top_llamacpp(choice):
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    if content and isinstance(content[0], dict):
        return content[0].get("top_logprobs") or []
    return []