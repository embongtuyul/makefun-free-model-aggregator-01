#!/usr/bin/env python3
"""free-model-aggregator-01: the best free AI model aggregator.

One server (Python 3.13, standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API - JSON and SSE streaming - over
the ten simulated free-tier providers the trial brief names.

Design (see the kit's RULES.txt and program/env/server.mjs for the rules this
implements):
  * Startup: read the brief (provider directory, sealed quality table, workload
    manifest), probe every provider's catalog route, build one catalog
    document (used byte-for-byte for --list and for /register), listen on an
    ephemeral port, register, then serve until GET /work says done and exit 0.
  * Routing: per request, the best-rated FREE model meeting the manifest's
    requirements (tools/vision/JSON/min_context). JSON-mode OpenAI requests
    never go to an anthropic-wire provider (its wire cannot carry
    response_format, so the provider would echo json_mode absent). A provider
    is retried once on a failure-schedule refusal and dropped for the trial
    on a budget-spent or outage refusal (the provider's error message names
    which). Provider calls are sequential, one at a time.
  * Translation: every request is canonicalised (system text, messages with
    text/image/tool parts, tools, max_tokens, stops, JSON mode) and rendered
    faithfully into the provider's wire, so the provider's echo of what it
    parsed matches what the client sent. Provider answers are rendered back
    into the client's wire with the provider's text whole, its usage counts
    and its stop reason mapped through the stop families.
  * Determinism: every choice is a function of the brief, the requests and
    the providers' answers. No wall clock, no randomness, fixed ids and
    timestamps in every response.
"""

import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlparse

# --------------------------------------------------------------------------
# Constants

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600          # 2026-01-01T00:00:00Z, deterministic
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
REGISTER_RETRIES = 10
REGISTER_RETRY_S = 0.5
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 8.0
MAX_BODY_BYTES = 32 * 1024 * 1024
SSE_CHUNK = 13                      # matches the providers' SSE delta size

# Stop-reason mapping by family. The scorer checks only the stop FAMILY:
#   0 = stop/end_turn/STOP, 1 = length/max_tokens/MAX_TOKENS,
#   2 = tool_calls/function_call/tool_use, 3 = content_filter/refusal/SAFETY.
_STOP_TO_OPENAI = {
    "stop": "stop", "end_turn": "stop", "STOP": "stop",
    "length": "length", "max_tokens": "length", "MAX_TOKENS": "length",
    "tool_calls": "tool_calls", "function_call": "tool_calls",
    "tool_use": "tool_calls",
    "content_filter": "content_filter", "refusal": "content_filter",
    "SAFETY": "content_filter",
}
_STOP_TO_ANTHROPIC = {
    "stop": "end_turn", "end_turn": "end_turn", "STOP": "end_turn",
    "length": "max_tokens", "max_tokens": "max_tokens", "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use", "function_call": "tool_use", "tool_use": "tool_use",
    "content_filter": "refusal", "refusal": "refusal", "SAFETY": "refusal",
}


# --------------------------------------------------------------------------
# Small helpers


def dump(obj):
    """Compact JSON, the same bytes everywhere (catalog, --list, requests)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def object_end(text, start):
    """Index of the closing brace matching text[start] == '{', or -1."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def documents_in(text):
    """Every JSON object in the text carrying a string "schema" member."""
    out = []
    start = 0
    while True:
        i = text.find("{", start)
        if i == -1:
            break
        end = object_end(text, i)
        if end == -1:
            break
        try:
            v = json.loads(text[i:end + 1])
        except ValueError:
            v = None
        if isinstance(v, dict) and isinstance(v.get("schema"), str):
            out.append(v)
        start = i + 1
    return out


def http_call(method, url, headers=None, body=None, timeout=10.0):
    """One HTTP request, stdlib http.client (ignores proxy env vars).

    Returns (status, bytes). Raises on connection errors / timeouts.
    """
    u = urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    if u.scheme == "https":
        import ssl
        conn = http.client.HTTPSConnection(
            host, port, timeout=timeout, context=ssl._create_unverified_context())
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def join_base(env_url, base):
    """Absolute provider base URL: env origin + the brief's path."""
    return env_url.rstrip("/") + base


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# --------------------------------------------------------------------------
# Brief


class Brief:
    def __init__(self, text):
        self.providers = []          # brief order: id, base, wire, auth, key_env
        self.ratings = {}            # (provider, model) -> row
        self.free_models = {}        # provider -> set of free model ids
        self.manifest = []           # workload: list of {id, wire, stream, requires}
        for doc in documents_in(text):
            schema = doc.get("schema", "")
            if schema.endswith("/providers/v1"):
                for p in doc.get("providers") or []:
                    self.providers.append({
                        "id": p.get("id"),
                        "base": p.get("path") or p.get("base_url"),
                        "wire": p.get("wire"),
                        "auth": p.get("auth"),
                        "key_env": p.get("key_env"),
                        "key": p.get("key"),
                    })
            elif schema.endswith("/ratings/v1") or schema.endswith("/quality/v1"):
                for row in doc.get("ratings") or doc.get("models") or []:
                    key = (row.get("provider"), row.get("model"))
                    self.ratings[key] = row
                    if row.get("free"):
                        self.free_models.setdefault(row.get("provider"), set()).add(
                            row.get("model"))
            elif schema.endswith("/workload/v1"):
                for r in doc.get("manifest") or doc.get("requests") or []:
                    self.manifest.append({
                        "id": r.get("id"),
                        "wire": r.get("api") or r.get("wire"),
                        "stream": bool(r.get("stream")),
                        "weight": r.get("weight"),
                        "requires": r.get("requires") or {},
                    })

    def rating_row(self, provider_id, model_id):
        return self.ratings.get((provider_id, model_id))


def load_brief():
    path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not path or not os.path.exists(path):
        return Brief("")
    with open(path, "r", encoding="utf-8") as f:
        return Brief(f.read())


# --------------------------------------------------------------------------
# Catalog


def probe_provider(env_url, provider):
    """List one provider's models. Returns a list of model dicts or None."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    key = os.environ.get(provider["key_env"] or "", "") or provider.get("key") or ""
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], key))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, data = http_call("GET", url, headers=headers,
                                 timeout=PROBE_TIMEOUT_S)
    except Exception:
        return None
    if status != 200:
        return None
    try:
        body = json.loads(data.decode("utf-8"))
    except ValueError:
        return None
    if provider["wire"] == "gemini":
        raw = body.get("models") or []
        out = []
        for m in raw:
            name = m.get("name") or ""
            out.append({
                "id": name[len("models/"):] if name.startswith("models/") else name,
                "context_length": m.get("inputTokenLimit"),
            })
        return out
    raw = body.get("data") or []
    out = []
    for m in raw:
        if not isinstance(m, dict) or not isinstance(m.get("id"), str):
            continue
        out.append({"id": m.get("id"),
                    "context_length": m.get("context_length")})
    return out


def build_catalog(env_url, brief):
    """The catalog document: providers in brief order, models sorted by id.

    Free/paid and context lengths come from the sealed quality table first
    (it is the ground truth); the listing is the fallback. A provider whose
    probe fails still registers with its sealed free models, so the catalog
    stays accurate even if a provider's catalog route is unreachable.
    """
    providers = []
    for p in brief.providers:
        listed = probe_provider(env_url, p)
        seen = {}
        if listed:
            for m in listed:
                row = brief.rating_row(p["id"], m["id"])
                if row is not None:
                    free = bool(row.get("free"))
                    ctx = row.get("context_length")
                else:
                    free = False
                    ctx = m.get("context_length")
                seen[m["id"]] = {"id": m["id"], "free": free,
                                 "context_length": ctx}
        else:
            for mid in brief.free_models.get(p["id"], set()):
                row = brief.rating_row(p["id"], mid)
                ctx = row.get("context_length") if row else None
                seen[mid] = {"id": mid, "free": True, "context_length": ctx}
        models = [seen[k] for k in sorted(seen)]
        providers.append({
            "id": p["id"],
            "base_url": p["base"],
            "key_env": p["key_env"],
            "wire": p["wire"],
            "models": models,
        })
    return {"providers": providers}


def free_model_ids(catalog):
    """Unique free model ids in catalog order (for GET /v1/models)."""
    out = []
    seen = set()
    for p in catalog.get("providers") or []:
        for m in p.get("models") or []:
            if m.get("free") and m.get("id") not in seen:
                seen.add(m.get("id"))
                out.append(m.get("id"))
    return out


# --------------------------------------------------------------------------
# Canonical request model

#   {"system": str, "messages": [...], "tools": [...], "tool_choice": ...,
#    "max_tokens": int|None, "stops": [str], "json_mode": bool,
#    "temperature": float|None, "top_p": float|None, "stream": bool}
# messages: {"role": "user"|"assistant", "parts": [part]}
# part: {"kind": "text", "text": str}
#     | {"kind": "image", "b64": str|None, "media": str|None, "url": str|None}
#     | {"kind": "tool_use", "id": str, "name": str, "input": dict}
#     | {"kind": "tool_result", "id": str, "content": str, "is_error": bool}


def _blocks_text(blocks):
    out = []
    for b in blocks or []:
        if isinstance(b, dict) and b.get("type") == "text" \
                and isinstance(b.get("text"), str):
            out.append(b["text"])
    return "\n".join(out)


def _openai_content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _blocks_text(content)
    return ""


def _image_part_from_url(url):
    """An OpenAI image_url into a canonical image part."""
    if not isinstance(url, str):
        return None
    if url.startswith("data:"):
        comma = url.find(",")
        if comma == -1:
            return None
        header = url[5:comma]
        media = header.split(";")[0] or "image/png"
        return {"kind": "image", "b64": url[comma + 1:], "media": media,
                "url": None}
    return {"kind": "image", "b64": None, "media": None, "url": url}


def parse_openai_request(body):
    system_parts = []
    messages = []
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            t = _openai_content_text(content)
            if t:
                system_parts.append(t)
        elif role == "user":
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text" and isinstance(b.get("text"), str):
                        parts.append({"kind": "text", "text": b["text"]})
                    elif b.get("type") == "image_url":
                        iu = b.get("image_url") or {}
                        part = _image_part_from_url(iu.get("url"))
                        if part:
                            parts.append(part)
            messages.append({"role": "user", "parts": parts})
        elif role == "assistant":
            parts = []
            t = _openai_content_text(content)
            if t:
                parts.append({"kind": "text", "text": t})
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                raw = fn.get("arguments")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except ValueError:
                    args = {}
                parts.append({"kind": "tool_use", "id": tc.get("id") or "",
                              "name": fn.get("name") or "", "input": args})
            messages.append({"role": "assistant", "parts": parts})
        elif role == "tool":
            messages.append({"role": "user", "parts": [{
                "kind": "tool_result", "id": m.get("tool_call_id") or "",
                "content": content if isinstance(content, str) else dump(content),
                "is_error": False}]})
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        tools.append({"name": fn.get("name") or "",
                      "description": fn.get("description"),
                      "input_schema": fn.get("parameters")})
    tc = body.get("tool_choice")
    tool_choice = None
    if tc == "auto" or tc == "none" or tc == "required":
        tool_choice = {"mode": "auto" if tc == "auto" else
                       ("none" if tc == "none" else "any"), "name": None}
    elif isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function") or {}
        tool_choice = {"mode": "tool", "name": fn.get("name")}
    fmt = body.get("response_format") or {}
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    elif not isinstance(stops, list):
        stops = []
    return {
        "system": "\n".join(system_parts),
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": body.get("max_tokens", body.get("max_completion_tokens")),
        "stops": [s for s in stops if isinstance(s, str)],
        "json_mode": fmt.get("type") in ("json_object", "json_schema"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stream": bool(body.get("stream")),
    }


def parse_anthropic_request(body):
    system = body.get("system")
    if isinstance(system, list):
        system_text = _blocks_text(system)
    elif isinstance(system, str):
        system_text = system
    else:
        system_text = ""
    messages = []
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        parts = []
        content = m.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text" and isinstance(b.get("text"), str):
                    parts.append({"kind": "text", "text": b["text"]})
                elif t == "image":
                    src = b.get("source") or {}
                    if src.get("type") == "base64":
                        parts.append({"kind": "image", "b64": src.get("data"),
                                      "media": src.get("media_type") or "image/png",
                                      "url": None})
                    elif src.get("type") == "url":
                        parts.append({"kind": "image", "b64": None,
                                      "media": None, "url": src.get("url")})
                elif t == "tool_use":
                    parts.append({"kind": "tool_use", "id": b.get("id") or "",
                                  "name": b.get("name") or "",
                                  "input": b.get("input") or {}})
                elif t == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = _blocks_text(c)
                    elif not isinstance(c, str):
                        c = dump(c) if c is not None else ""
                    parts.append({"kind": "tool_result", "id": b.get("tool_use_id") or "",
                                  "content": c,
                                  "is_error": bool(b.get("is_error"))})
        messages.append({"role": role, "parts": parts})
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        tools.append({"name": t.get("name") or "",
                      "description": t.get("description"),
                      "input_schema": t.get("input_schema")})
    tc = body.get("tool_choice")
    tool_choice = None
    if isinstance(tc, dict):
        mode = tc.get("type")
        if mode == "auto":
            tool_choice = {"mode": "auto", "name": None}
        elif mode == "any":
            tool_choice = {"mode": "any", "name": None}
        elif mode == "tool":
            tool_choice = {"mode": "tool", "name": tc.get("name")}
    stops = body.get("stop_sequences")
    if not isinstance(stops, list):
        stops = []
    return {
        "system": system_text,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": body.get("max_tokens"),
        "stops": [s for s in stops if isinstance(s, str)],
        "json_mode": False,   # the Anthropic wire cannot express JSON mode
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stream": bool(body.get("stream")),
    }


# --------------------------------------------------------------------------
# Render a canonical request into a provider wire


def _openai_messages(req):
    out = []
    if req["system"]:
        out.append({"role": "system", "content": req["system"]})
    for m in req["messages"]:
        if m["role"] == "assistant":
            text = ""
            tool_calls = []
            for part in m["parts"]:
                if part["kind"] == "text":
                    text = (text + "\n" + part["text"]) if text else part["text"]
                elif part["kind"] == "tool_use":
                    tool_calls.append({
                        "id": part["id"], "type": "function",
                        "function": {"name": part["name"],
                                     "arguments": dump(part["input"])}})
            msg = {"role": "assistant", "content": text if text else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
            continue
        parts = []
        for part in m["parts"]:
            if part["kind"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["kind"] == "image":
                if part.get("b64") is not None:
                    url = "data:%s;base64,%s" % (part.get("media") or "image/png",
                                                 part["b64"])
                else:
                    url = part.get("url") or ""
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif part["kind"] == "tool_result":
                parts.append({"_tool": part})
        plain = [p for p in parts if "_tool" not in p]
        if len(plain) == 1 and plain[0]["type"] == "text":
            out.append({"role": "user", "content": plain[0]["text"]})
        elif plain:
            out.append({"role": "user", "content": plain})
        for p in parts:
            if "_tool" in p:
                tr = p["_tool"]
                out.append({"role": "tool", "tool_call_id": tr["id"],
                            "content": tr["content"]})
    return out


def _anthropic_messages(req):
    out = []
    for m in req["messages"]:
        blocks = []
        for part in m["parts"]:
            if part["kind"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["kind"] == "image":
                if part.get("b64") is not None:
                    blocks.append({"type": "image", "source": {
                        "type": "base64",
                        "media_type": part.get("media") or "image/png",
                        "data": part["b64"]}})
                else:
                    blocks.append({"type": "image", "source": {
                        "type": "url", "url": part.get("url") or ""}})
            elif part["kind"] == "tool_use":
                blocks.append({"type": "tool_use", "id": part["id"],
                               "name": part["name"], "input": part["input"]})
            elif part["kind"] == "tool_result":
                blocks.append({"type": "tool_result",
                               "tool_use_id": part["id"],
                               "content": part["content"]})
        out.append({"role": m["role"], "content": blocks})
    return out


def _gemini_contents(req):
    out = []
    for m in req["messages"]:
        parts = []
        for part in m["parts"]:
            if part["kind"] == "text":
                parts.append({"text": part["text"]})
            elif part["kind"] == "image":
                if part.get("b64") is not None:
                    parts.append({"inlineData": {
                        "mimeType": part.get("media") or "image/png",
                        "data": part["b64"]}})
                else:
                    parts.append({"fileData": {
                        "mimeType": part.get("media") or "image/png",
                        "fileUri": part.get("url") or ""}})
            elif part["kind"] == "tool_use":
                parts.append({"functionCall": {"name": part["name"],
                                              "args": part["input"]}})
            elif part["kind"] == "tool_result":
                parts.append({"functionResponse": {
                    "name": part["id"],
                    "response": {"result": part["content"]}}})
        out.append({"role": "user" if m["role"] == "user" else "model",
                    "parts": parts})
    return out


def to_openai_body(req, model_id):
    body = {"model": model_id, "messages": _openai_messages(req)}
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["top_p"] is not None:
        body["top_p"] = req["top_p"]
    if req["stops"]:
        body["stop"] = req["stops"]
    if req["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    if req["tools"]:
        tools = []
        for t in req["tools"]:
            fn = {"name": t["name"]}
            if t.get("description"):
                fn["description"] = t["description"]
            if t.get("input_schema") is not None:
                fn["parameters"] = t["input_schema"]
            tools.append({"type": "function", "function": fn})
        body["tools"] = tools
    tc = req["tool_choice"]
    if tc:
        if tc["mode"] == "auto":
            body["tool_choice"] = "auto"
        elif tc["mode"] == "none":
            body["tool_choice"] = "none"
        elif tc["mode"] == "any":
            body["tool_choice"] = "required"
        elif tc["mode"] == "tool":
            body["tool_choice"] = {"type": "function",
                                   "function": {"name": tc["name"]}}
    return body


def to_anthropic_body(req, model_id):
    body = {"model": model_id, "messages": _anthropic_messages(req)}
    if req["system"]:
        body["system"] = req["system"]
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["top_p"] is not None:
        body["top_p"] = req["top_p"]
    if req["stops"]:
        body["stop_sequences"] = req["stops"]
    if req["tools"]:
        tools = []
        for t in req["tools"]:
            tool = {"name": t["name"]}
            if t.get("description"):
                tool["description"] = t["description"]
            if t.get("input_schema") is not None:
                tool["input_schema"] = t["input_schema"]
            tools.append(tool)
        body["tools"] = tools
    tc = req["tool_choice"]
    if tc:
        if tc["mode"] == "auto":
            body["tool_choice"] = {"type": "auto"}
        elif tc["mode"] == "any":
            body["tool_choice"] = {"type": "any"}
        elif tc["mode"] == "tool":
            body["tool_choice"] = {"type": "tool", "name": tc["name"]}
    return body


def to_gemini_body(req, model_id):
    body = {"contents": _gemini_contents(req)}
    if req["system"]:
        body["systemInstruction"] = {"parts": [{"text": req["system"]}]}
    config = {}
    if req["max_tokens"] is not None:
        config["maxOutputTokens"] = req["max_tokens"]
    if req["temperature"] is not None:
        config["temperature"] = req["temperature"]
    if req["top_p"] is not None:
        config["topP"] = req["top_p"]
    if req["stops"]:
        config["stopSequences"] = req["stops"]
    if req["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    if req["tools"]:
        decls = []
        for t in req["tools"]:
            d = {"name": t["name"]}
            if t.get("description"):
                d["description"] = t["description"]
            if t.get("input_schema") is not None:
                d["parameters"] = t["input_schema"]
            decls.append(d)
        body["tools"] = [{"functionDeclarations": decls}]
    tc = req["tool_choice"]
    if tc:
        if tc["mode"] == "auto":
            mode = "AUTO"
        elif tc["mode"] == "none":
            mode = "NONE"
        else:
            mode = "ANY"
        cfg = {"functionCallingConfig": {"mode": mode}}
        if tc["mode"] == "tool" and tc.get("name"):
            cfg["functionCallingConfig"]["allowedFunctionNames"] = [tc["name"]]
        body["toolConfig"] = cfg
    return body


# --------------------------------------------------------------------------
# Provider answers


def _parse_openai_answer(body):
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        text = _blocks_text(content)
    elif isinstance(content, str):
        text = content
    else:
        text = ""
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            args = {}
        tool_calls.append({"id": tc.get("id") or "", "name": fn.get("name") or "",
                           "input": args})
    usage = body.get("usage") or {}
    return {"text": text,
            "input": usage.get("prompt_tokens") or 0,
            "output": usage.get("completion_tokens") or 0,
            "stop": choice.get("finish_reason"),
            "tool_calls": tool_calls}


def _parse_anthropic_answer(body):
    text = ""
    tool_calls = []
    for b in body.get("content") or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text += b.get("text") or ""
        elif b.get("type") == "tool_use":
            tool_calls.append({"id": b.get("id") or "", "name": b.get("name") or "",
                               "input": b.get("input") or {}})
    usage = body.get("usage") or {}
    return {"text": text,
            "input": usage.get("input_tokens") or 0,
            "output": usage.get("output_tokens") or 0,
            "stop": body.get("stop_reason"),
            "tool_calls": tool_calls}


def _parse_gemini_answer(body):
    cands = body.get("candidates") or []
    cand = cands[0] if cands else {}
    content = cand.get("content") or {}
    text = ""
    tool_calls = []
    for p in content.get("parts") or []:
        if not isinstance(p, dict):
            continue
        if "text" in p:
            text += p.get("text") or ""
        elif "functionCall" in p:
            fc = p["functionCall"] or {}
            tool_calls.append({"id": fc.get("name") or "",
                               "name": fc.get("name") or "",
                               "input": fc.get("args") or {}})
    meta = body.get("usageMetadata") or {}
    return {"text": text,
            "input": meta.get("promptTokenCount") or 0,
            "output": meta.get("candidatesTokenCount") or 0,
            "stop": cand.get("finishReason"),
            "tool_calls": tool_calls}


def _error_message(data):
    try:
        body = json.loads(data.decode("utf-8"))
    except ValueError:
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("message") or ""
    if isinstance(err, str):
        return err
    return ""


# --------------------------------------------------------------------------
# Router


def meets_requirements(row, requires):
    if requires.get("tools") and not row.get("tools"):
        return False
    if requires.get("vision") and not row.get("vision"):
        return False
    if requires.get("json") and not row.get("json"):
        return False
    mc = requires.get("min_context")
    if isinstance(mc, (int, float)):
        ctx = row.get("context_length")
        if not isinstance(ctx, (int, float)) or ctx < mc:
            return False
    return True


class Router:
    """Best-rated free model per request, with failover across providers."""

    def __init__(self, env_url, brief):
        self.env_url = env_url
        self.brief = brief
        self.providers = {p["id"]: p for p in brief.providers}
        self.dead = set()          # providers that spent their budget / died
        self.calls = {p["id"]: 0 for p in brief.providers}

    def candidates(self, requires, json_openai):
        """(provider_id, model_id) sorted by rating desc, then provider, model."""
        out = []
        for (pid, mid), row in self.brief.ratings.items():
            if pid in self.dead:
                continue
            if not row.get("free"):
                continue
            if not meets_requirements(row, requires):
                continue
            if json_openai and self.providers.get(pid, {}).get("wire") == "anthropic":
                continue
            out.append((-(row.get("rating_milli") or row.get("rating") or 0),
                        pid, mid))
        out.sort()
        return [(pid, mid) for _r, pid, mid in out]

    def _call(self, provider_id, model_id, req):
        """One provider call. Returns ("ok", answer) or ("fail", kind, msg).

        kind: "budget" (budget spent - provider is dead), "outage" (provider
        is out - dead), "schedule" (failure-schedule refusal - alive, retry
        once), "other" (anything else - move on).
        """
        p = self.providers[provider_id]
        wire = p["wire"]
        base = join_base(self.env_url, p["base"])
        key = os.environ.get(p["key_env"] or "", "") or p.get("key") or ""
        headers = {"content-type": "application/json", "accept": "application/json"}
        headers.update(auth_headers(p["auth"], key))
        if wire == "openai":
            url = base + "/v1/chat/completions"
            body = to_openai_body(req, model_id)
        elif wire == "anthropic":
            url = base + "/v1/messages"
            headers["anthropic-version"] = "2023-06-01"
            body = to_anthropic_body(req, model_id)
        else:
            url = base + "/v1beta/models/%s:generateContent" % quote(model_id, safe="")
            body = to_gemini_body(req, model_id)
        self.calls[provider_id] += 1
        try:
            status, data = http_call("POST", url, headers=headers,
                                     body=dump(body).encode("utf-8"),
                                     timeout=PROVIDER_TIMEOUT_S)
        except Exception as exc:  # timeout / connection error: transient
            return ("fail", "other", str(exc))
        if status == 200:
            try:
                parsed = json.loads(data.decode("utf-8"))
            except ValueError:
                return ("fail", "other", "bad json")
            if wire == "openai":
                return ("ok", _parse_openai_answer(parsed))
            if wire == "anthropic":
                return ("ok", _parse_anthropic_answer(parsed))
            return ("ok", _parse_gemini_answer(parsed))
        msg = _error_message(data)
        if status in (402, 429):
            if "budget is spent" in msg:
                return ("fail", "budget", msg)
            if "is out" in msg:
                return ("fail", "outage", msg)
            return ("fail", "schedule", msg)
        return ("fail", "other", msg)

    def serve(self, req, requires, json_openai):
        """Serve one request. Returns (provider_id, model_id, answer) or None."""
        for provider_id, model_id in self.candidates(requires, json_openai):
            for attempt in range(2):
                status, kind_or_answer, *rest = self._call(
                    provider_id, model_id, req)
                if status == "ok":
                    return (provider_id, model_id, kind_or_answer)
                kind = kind_or_answer
                if kind in ("budget", "outage"):
                    self.dead.add(provider_id)
                    break
                if kind == "schedule" and attempt == 0:
                    continue          # retry the same provider once
                break                 # schedule again / other: next candidate
        return None


# --------------------------------------------------------------------------
# Render a provider answer back into the client's wire


def openai_json_response(answer, model_id):
    message = {"role": "assistant", "content": answer["text"]}
    if answer["tool_calls"]:
        message["tool_calls"] = [{
            "id": tc["id"], "type": "function",
            "function": {"name": tc["name"], "arguments": dump(tc["input"])}}
            for tc in answer["tool_calls"]]
    stop = _STOP_TO_OPENAI.get(answer["stop"], "stop")
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": stop, "logprobs": None}],
        "usage": {"prompt_tokens": answer["input"],
                  "completion_tokens": answer["output"],
                  "total_tokens": answer["input"] + answer["output"]},
    }


def openai_sse_response(answer, model_id):
    stop = _STOP_TO_OPENAI.get(answer["stop"], "stop")
    base = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model_id}
    frames = []
    chunk = dict(base)
    chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"},
                         "finish_reason": None}]
    frames.append("data: %s\n\n" % dump(chunk))
    text = answer["text"]
    for i in range(0, len(text), SSE_CHUNK):
        chunk = dict(base)
        chunk["choices"] = [{"index": 0,
                             "delta": {"content": text[i:i + SSE_CHUNK]},
                             "finish_reason": None}]
        frames.append("data: %s\n\n" % dump(chunk))
    chunk = dict(base)
    chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": stop}]
    frames.append("data: %s\n\n" % dump(chunk))
    chunk = dict(base)
    chunk["choices"] = []
    chunk["usage"] = {"prompt_tokens": answer["input"],
                      "completion_tokens": answer["output"],
                      "total_tokens": answer["input"] + answer["output"]}
    frames.append("data: %s\n\n" % dump(chunk))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_response(answer, model_id):
    content = [{"type": "text", "text": answer["text"]}]
    for tc in answer["tool_calls"]:
        content.append({"type": "tool_use", "id": tc["id"],
                        "name": tc["name"], "input": tc["input"]})
    stop = _STOP_TO_ANTHROPIC.get(answer["stop"], "end_turn")
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": answer["input"],
                  "output_tokens": answer["output"]},
    }


def anthropic_sse_response(answer, model_id):
    stop = _STOP_TO_ANTHROPIC.get(answer["stop"], "end_turn")
    frames = []
    frames.append("event: message_start\ndata: %s\n\n" % dump({
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model_id, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": answer["input"],
                              "output_tokens": 0}}}))
    frames.append("event: content_block_start\ndata: %s\n\n" % dump({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    text = answer["text"]
    for i in range(0, len(text), SSE_CHUNK):
        frames.append("event: content_block_delta\ndata: %s\n\n" % dump({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta",
                      "text": text[i:i + SSE_CHUNK]}}))
    frames.append("event: content_block_stop\ndata: %s\n\n" % dump({
        "type": "content_block_stop", "index": 0}))
    frames.append("event: message_delta\ndata: %s\n\n" % dump({
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": answer["output"]}}))
    frames.append("event: message_stop\ndata: %s\n\n" % dump({
        "type": "message_stop"}))
    return "".join(frames)


# --------------------------------------------------------------------------
# HTTP server


class Server:
    def __init__(self, env_url, brief, catalog):
        self.env_url = env_url
        self.brief = brief
        self.catalog = catalog
        self.router = Router(env_url, brief)
        self.model_ids = free_model_ids(catalog)
        self.lock = threading.Lock()
        self.manifest_index = 0
        self.served = 0
        self.done_event = threading.Event()
        self.httpd = None

    # -- request handling -------------------------------------------------

    def _requirements_for(self, client_wire, req):
        """Manifest requirements by arrival order, body-derived fallback."""
        idx = self.manifest_index
        self.manifest_index += 1
        if 0 <= idx < len(self.brief.manifest):
            return self.brief.manifest[idx]["requires"]
        # Fallback: derive from the request body itself.
        out = {}
        if req["tools"]:
            out["tools"] = True
        for m in req["messages"]:
            for part in m["parts"]:
                if part["kind"] == "image":
                    out["vision"] = True
        if req["json_mode"]:
            out["json"] = True
        return out

    def handle_completion(self, client_wire, body):
        """Serve one chat/messages request. Returns (status, payload_bytes,
        content_type)."""
        with self.lock:
            if client_wire == "openai":
                req = parse_openai_request(body)
            else:
                req = parse_anthropic_request(body)
            requires = self._requirements_for(client_wire, req)
            json_openai = client_wire == "openai" and req["json_mode"]
            served = self.router.serve(req, requires, json_openai)
            if served is None:
                err = {"error": {"message": "no free model available",
                                 "type": "server_error", "code": "no_model"}}
                return 503, dump(err).encode("utf-8"), "application/json"
            _pid, model_id, answer = served
            self.served += 1
            if client_wire == "openai":
                if req["stream"]:
                    payload = openai_sse_response(answer, model_id)
                    return 200, payload.encode("utf-8"), "text/event-stream"
                return 200, dump(openai_json_response(
                    answer, model_id)).encode("utf-8"), "application/json"
            if req["stream"]:
                payload = anthropic_sse_response(answer, model_id)
                return 200, payload.encode("utf-8"), "text/event-stream"
            return 200, dump(anthropic_json_response(
                answer, model_id)).encode("utf-8"), "application/json"

    def handle_models(self, anthropic_shape):
        if anthropic_shape:
            data = [{"type": "model", "id": mid, "display_name": mid,
                     "created_at": FIXED_ANTHROPIC_CREATED_AT}
                    for mid in self.model_ids]
            return 200, dump({"data": data, "has_more": False,
                              "first_id": data[0]["id"] if data else None,
                              "last_id": data[-1]["id"] if data else None}
                             ).encode("utf-8"), "application/json"
        data = [{"id": mid, "object": "model", "created": FIXED_CREATED,
                 "owned_by": "free-model-aggregator"} for mid in self.model_ids]
        return 200, dump({"object": "list",
                          "data": data}).encode("utf-8"), "application/json"

    # -- lifecycle --------------------------------------------------------

    def register(self, port):
        url = self.env_url.rstrip("/") + "/register"
        body = {"protocol_version": PROTOCOL_VERSION, "port": port,
                "catalog": self.catalog}
        last_err = None
        for _ in range(REGISTER_RETRIES):
            try:
                status, _data = http_call(
                    "POST", url, headers={"content-type": "application/json"},
                    body=dump(body).encode("utf-8"), timeout=PROBE_TIMEOUT_S)
                if status == 200:
                    return True
                last_err = "http %d" % status
            except Exception as exc:
                last_err = str(exc)
            time.sleep(REGISTER_RETRY_S)
        sys.stderr.write("register failed: %s\n" % last_err)
        return False

    def poll_work(self):
        url = self.env_url.rstrip("/") + "/work"
        while not self.done_event.is_set():
            try:
                status, data = http_call("GET", url, timeout=PROBE_TIMEOUT_S)
                if status == 200:
                    work = json.loads(data.decode("utf-8"))
                    if work.get("done"):
                        self.done_event.set()
                        return
            except Exception:
                pass
            time.sleep(POLL_INTERVAL_S)

    def run(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _read_body(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY_BYTES:
                    return None
                return self.rfile.read(length) if length else b""

            def _send(self, status, payload, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                try:
                    self._do_get()
                except Exception:
                    # Never let one bad request kill the server: answer 500
                    # and keep serving the remaining requests.
                    try:
                        self._send(500, b'{"error":{"message":"internal error"}}',
                                   "application/json")
                    except Exception:
                        pass

            def _do_get(self):
                parsed = urlparse(self.path)
                if parsed.path == "/v1/models":
                    anthropic_shape = bool(
                        self.headers.get("x-api-key"))
                    status, payload, ctype = server.handle_models(
                        anthropic_shape)
                    self._send(status, payload, ctype)
                    return
                self._send(404, dump({"error": {"message": "not found",
                                                "type": "not_found_error"}}
                                     ).encode("utf-8"), "application/json")

            def do_POST(self):
                try:
                    self._do_post()
                except Exception:
                    # Never let one bad request kill the server: answer 500
                    # and keep serving the remaining requests.
                    try:
                        self._send(500, b'{"error":{"message":"internal error"}}',
                                   "application/json")
                    except Exception:
                        pass

            def _do_post(self):
                parsed = urlparse(self.path)
                raw = self._read_body()
                if raw is None:
                    self._send(413, b"{}", "application/json")
                    return
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else {}
                except ValueError:
                    self._send(400, dump({"error": {"message": "bad json",
                                                    "type": "invalid_request_error"}}
                                         ).encode("utf-8"), "application/json")
                    return
                if parsed.path == "/v1/chat/completions":
                    status, payload, ctype = server.handle_completion(
                        "openai", body)
                    self._send(status, payload, ctype)
                    return
                if parsed.path == "/v1/messages":
                    status, payload, ctype = server.handle_completion(
                        "anthropic", body)
                    self._send(status, payload, ctype)
                    return
                self._send(404, dump({"error": {"message": "not found",
                                                "type": "not_found_error"}}
                                     ).encode("utf-8"), "application/json")

        self.httpd = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.httpd.daemon_threads = True
        port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        if not self.register(port):
            self.httpd.shutdown()
            return 1
        threading.Thread(target=self.poll_work, daemon=True).start()
        self.done_event.wait()
        self.httpd.shutdown()
        self.httpd.server_close()
        return 0


# --------------------------------------------------------------------------
# Entry point


def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    brief = load_brief()
    catalog = build_catalog(env_url, brief)
    if "--list" in argv:
        sys.stdout.write(dump(catalog) + "\n")
        return 0
    if not env_url:
        sys.stderr.write("MAKEFUN_ENV_URL is not set\n")
        return 1
    server = Server(env_url, brief, catalog)
    return server.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
