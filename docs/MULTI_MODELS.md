# Serve several models on demand

`strata serve-models` exposes several text models through one API. It loads the model selected by a request's
`model` field and keeps at most one model resident. This is an opt-in server mode; ordinary `strata serve` is unchanged.

Create a router config that maps public model IDs to existing Strata run configs:

```json
{
  "host": "127.0.0.1",
  "port": 8081,
  "models": {
    "stock": "strata-stock.json",
    "abliterated": "strata-abliterated.json"
  }
}
```

Paths to run configs are relative to the router config. Engine, tokenizer, model and other paths inside a run config
keep their ordinary Strata meaning; use absolute paths when moving a setup between checkouts.

```sh
strata serve-models --config models.json
```

From a source checkout, use `python strata.py serve-models --config models.json`. The model IDs in this file are the
names listed by `GET /v1/models` and accepted by requests. Original run-config model names and aliases also remain
accepted and appear in each model's `aliases` list. An ambiguous name shared by two models is rejected at startup.

```sh
curl http://127.0.0.1:8081/v1/models
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"stock","messages":[{"role":"user","content":"Hello"}]}'
```

## What happens on a request

No engine starts when the router starts. The first request starts the selected model's private Strata server and
loads its engine. Requests for the resident model reuse that process. A request for a different model waits until
the current response ends, unloads and stops the old server, and starts the new one. Streamed responses hold their
turn until the backend stream ends; switching never interrupts a connected client's generation.

Requests run one at a time in arrival order after their bodies have been read. Up to 32 calls can wait (set
`"max_queue": 0` to reject waiting calls, or another whole number up to 4096). A full queue returns HTTP 429.
Disconnected queued clients are removed without loading their model. A disconnected active client cancels the
backend request. Switching waits for cancellation or stops the old process before starting its replacement.

This saves memory, but switching incurs model startup time and discards the old engine's KV/SSM and conversation
caches. Clients must send their conversation history, as with the ordinary stateless API. There is no concurrent
inference across models, no shared tensor allocation, and no change to quantization or model weights.

Each backend keeps its run config's sampling and thinking defaults and a snapshot of its adjacent
`.shared-settings.json` taken when it starts. Request overrides, including reasoning effort and uncapped token values,
are forwarded unchanged. Original configs and shared settings are never written. The router controls residency;
per-model idle-unload timers are disabled in its private config snapshots.

## API and access

Supported inference routes are `/v1/chat/completions`, `/v1/responses`, `/v1/messages` and
`/v1/messages/count_tokens`, including streaming and tool calls handled by the original Strata server.
Every request must name a configured model. Unknown or omitted IDs return HTTP 404 without loading anything.

`POST /v1/load` and `POST /v1/unload` accept `{"model":"stock"}` and require JSON. Loading follows the same
serialized switching policy; unloading an already unloaded model does nothing. `GET /v1/status` reports the active
model, queue length and last model startup time. `/health` reports API availability, including when no model is loaded.

Set `api_key` in the router config or `STRATA_API_KEY` to protect the public API (Bearer or `x-api-key`). The router
reuses Strata's Host and browser-origin checks and supports `allowed_hosts`, `cors_origins` and `trusted_origins`.
Only the router credential is public; each child binds a random loopback port with a private generated credential.
Use an API key when listening beyond localhost, and keep this behind HTTPS when accessing it remotely.

This mode exposes the APIs above, not the single-model web app, settings editor, monitor telemetry or slot files.
Backend logs appear in the router's output and in each run config's configured engine log. Failed starts return 503;
a later request can retry or select another model. Stopping the router cancels active requests and closes its child.
Backends and their engines use Strata's existing parent-death protection.
With systemd, use `KillMode=mixed` so the router coordinates the graceful stop; systemd can still kill its whole
control group if the stop deadline expires.

## Validation

`python -m unittest serve.test_model_router` runs without a GPU. It covers model selection and process reuse,
stream forwarding, protocol routes, exact request forwarding, FIFO/overload behavior, disconnects, access checks,
failed starts and recovery. Real-model switching times depend on storage, OS file cache and the selected run configs.
