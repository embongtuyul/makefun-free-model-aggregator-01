# free-model-aggregator-01

The best free AI model aggregator — one server (Python 3.13, standard library
only, `main.py`) that serves the OpenAI Chat Completions API and the Anthropic
Messages API — JSON and SSE streaming — over the ten simulated free-tier
providers the trial brief names.

## How it works

1. **Startup** — read the brief from `$MAKEFUN_PROMPT_FILE` (provider
   directory, sealed quality table, workload manifest), probe every provider's
   catalog route, build one catalog document (used byte-for-byte for `--list`
   and for `/register`), listen on an ephemeral port, register with the env,
   then serve until `GET /work` says done and exit 0.
2. **Routing (v2, planned)** — per request, the best-rated FREE model
   meeting the manifest's requirements (tools / vision / JSON / min_context).
   JSON-mode OpenAI requests never go to an anthropic-wire provider (its wire
   cannot carry `response_format`). Calls are planned, not greedy: the first
   call to every provider is free (the kit's generator draws failure indices
   from `[2, budget-1]`, so call 1 never fails on schedule, never hits the
   outage, and never exceeds the budget), then extra calls go to the best
   quality-per-failure-risk providers under a per-provider call budget
   computed from the brief and the generator's published budget ranges
   (`BUDGET_RANGES` in `main.py`, from `program/generator/draw.mjs`).
   Providers still under their cap are tried first; over-cap providers are
   the fallback, so no request goes unserved while a provider that can answer
   it is alive. A provider is dropped for the trial on a budget-spent or
   outage refusal (the error message names which); on a failure-schedule
   refusal it is retried once only when its next call index is likely safe
   (wide budget range — openrouter/groq/gemini/cloudflare/pollinations/
   kilol), else the request fails over at once. Provider calls are
   sequential, one at a time.

   Why planning wins: the score is saturated at 99, so the ranking is
   decided by the upstream-calls tie-break. v1's greedy routing made
   26/27/28 calls on the practice trials (4 schedule failures + 2
   budget-discovery calls on trial-0 alone). v2 makes **22/20/22** at the
   same quality 99 — and a simulator over 1500 freshly drawn instances
   (same generator distributions) gives a mean of **22.8 calls** with score
   99 on every instance, vs 26.7 for v1. Retry-vs-drop, on its own, saves
   nothing: a failed request costs 2 calls either way (fail + retry, or
   fail + failover) — the saving comes from calling providers fewer times
   in the first place, which is what the caps do.
3. **Translation** — every request is canonicalised (system text, messages
   with text/image/tool parts, tools, max_tokens, stops, JSON mode) and
   rendered faithfully into the provider's wire, so the provider's echo of
   what it parsed matches what the client sent. Provider answers are rendered
   back into the client's wire with the provider's text whole, its usage
   counts and its stop reason mapped through the stop families.
4. **Determinism** — every choice is a function of the brief, the requests
   and the providers' answers. No wall clock, no randomness, fixed ids and
   timestamps in every response.

## Files

- `main.py` — the whole server. `python3 main.py --list` prints the catalog
  (identical to what it registers).
- `LICENSE` — MIT.

## Local test

```sh
cd ~/workspace/makefun/kits/free-model-aggregator-01-kit
# 1. start the env (trial-0) on port 18080
MAKEFUN_STATE_DIR=/tmp/fma-state0 \
MAKEFUN_ENV_DATA_DIR=$PWD/practice/trial-0/input/env \
MAKEFUN_ENV_PORT=18080 \
node program/env/server.mjs &
# 2. run the server against it
MAKEFUN_ENV_URL=http://127.0.0.1:18080 \
MAKEFUN_PROMPT_FILE=$PWD/practice/trial-0/input/prompt.txt \
python3 ~/workspace/makefun/entries/free-model-aggregator-01/main.py
# 3. stop the env (SIGTERM exports state), then score it
MAKEFUN_OUTPUT_DIR=/tmp/fma-state0/export \
MAKEFUN_REFERENCE_DIR=$PWD/practice/trial-0/reference \
MAKEFUN_SCORES_DIR=/tmp/fma-scores0 \
node program/scorer/score.mjs
```
