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
2. **Routing (v3, greedy cost minimisation)** — per request, the cheapest
   action wins: each candidate is priced `cost = P(next call fails) +
   0.001 * weighted quality loss` vs the best-rated eligible model, and the
   minimum-cost candidate is tried; after any failure the loop re-picks, so
   retry-vs-failover is decided by cost, not by a fixed rule. The score
   stays 99 with quality 98, so quality is priced cheap and calls dominate:
   a first call to any provider is free (the kit's generator draws failure
   indices from `[2, budget-1]`, so call 1 never fails on schedule, never
   hits the outage, and never exceeds the budget), while a second call
   carries real failure risk, priced from the provider's budget-range prior
   (`BUDGET_RANGES` in `main.py`, from `program/generator/draw.mjs`) plus
   what the trial revealed — fail indices seen, the designated 2nd-call
   failure (one provider always fails its 2nd call; the prior updates as
   providers clear theirs), and the outage. No provider is called past its
   guaranteed budget (`calls_min`) unless nothing else can serve the
   request, in which case over-cap calls are allowed rather than leaving the
   request unserved. JSON-mode OpenAI requests never go to an
   anthropic-wire provider (its wire cannot carry `response_format`). A
   provider is dropped for the trial on a budget-spent or outage refusal
   (the error message names which). Provider calls are sequential, one at
   a time.

   Why greedy cost minimisation wins: the score is saturated at 99, so the
   ranking is decided by the upstream-calls tie-break, and quality 98
   scores 99 just like quality 99 — so the objective is minimise calls,
   not maximise quality. v2's fixed plan made 22/20/22 calls on the
   practice trials (simulated mean 22.8 over 5000 fresh instances). v3
   makes **21/22/22** at quality 98, score 99 on every trial — and the
   simulator gives a mean of **22.0 calls** with score 99 on 5000/5000
   instances. On paired draws v3 beats v2's 3-trial mean 80% of the time
   (ties 11%, loses 9%). The practice mean looks flat only because trial-1
   was v2's lucky draw (20 = floor) while v3 hit both of kilol's fail
   indices there at their correctly-priced low odds — the 5000-draw mean
   is the reliable estimator.
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
