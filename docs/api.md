# HTTP API and configuration

Start with `leatengine serve MODEL.gguf`. The default address is `http://127.0.0.1:8080`.
The API implements a subset of OpenAI chat completions and embeddings, plus model-management
and metrics extensions. It serves no HTML, frontend, or browser application.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /v1/models` | List configured chat models and their load status |
| `POST /v1/models/load` | Load a configured model, replacing the current chat model |
| `POST /v1/chat/completions` | Generate one reply, optionally streamed |
| `POST /v1/embeddings` | Embed text using the separately configured embedding model |
| `GET /metrics` | Prometheus metrics with the `leatengine_` prefix |

All routes use the same authentication when keys are enabled. Request bodies are limited to
64 MiB. Errors use an `error` object
with `message`, `type`, `param`, and `code`. A generation error after streaming starts arrives
as an SSE error event; clients must handle that as well as HTTP errors.

## Models

`serve` accepts files or directories containing GGUFs. Model IDs are filename stems. The first
chat model loads at startup. Recognized projectors and MTP files are excluded from the list.

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/models/load \
  -H 'Content-Type: application/json' -d '{"model":"model-file-stem"}'
```

Entries have `status` (`loading`, `loaded`, or `unloaded`). A loaded model also reports
`max_context`, available `reasoning.efforts` and its `default`, sampling defaults for
`reasoning` and `plain` replies, and `vision`/`image_tokens` when images are supported.
Clients should read these capabilities rather than import engine internals.

Only one chat model is loaded at a time. The `model` field of a completion request does not
switch models: generation uses the currently loaded model. Use the explicit load endpoint.

## Chat completions

An OpenAI Python client can use `base_url="http://127.0.0.1:8080/v1"` and an API key (any
placeholder when authentication is disabled). The client library is not a runtime dependency.

Supported controls include `messages`, `stream`, `stream_options.include_usage`, `max_tokens`
or `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`seed`, `stop`, `tools`, and `tool_choice` (`auto` or `none`). Templates determine which
message roles and tool formats a model understands.

- Streaming uses Server-Sent Events: JSON `data:` records followed by `data: [DONE]` on success.
  Disconnecting cancels the completion and releases its slot.
- Reasoning is returned as `reasoning_content`. `reasoning_effort` and
  `chat_template_kwargs` control supported templates; explicit template options take precedence.
- Tool calls are returned in OpenAI-style `tool_calls`. The engine does not execute them.
  A reply cut off by `max_tokens` or the context keeps `finish_reason: "length"` even when it
  holds complete calls, as a later call may be missing.
- Image content parts accept `image_url` with base64 `data:` URLs. Remote image URLs are not
  fetched. Supply `--mmproj` or a recognized projector beside the model.
- `service_tier: "flex"` schedules a completion behind foreground requests.
- `safety_identifier`, or `user` when absent, isolates cached prefixes within an API key.
  Clients sharing a key must supply distinct identifiers when their prompt caches should be separate.
- Replies include usage and a `timings` extension. Cached token counts appear in
  `usage.prompt_tokens_details.cached_tokens`; streamed usage requires `include_usage`.

This is not a complete OpenAI implementation: multiple choices, log probabilities, constrained
JSON output, and nonzero frequency penalties are unsupported. Responses, legacy text completions,
audio, file-storage, and tool-execution endpoints are not implemented. A prompt must fit the
configured context; clients manage conversation history.

## Embeddings

Start with `--embed /path/to/embedding-model.gguf`. The endpoint accepts `input` as a string
or a nonempty list of strings and returns float vectors, indexes, model ID, and token usage.
It uses the configured embedding model, independently of the chat model. Without a configured
embedding model, the endpoint returns 404. Inputs are truncated to the embedding context;
model-specific query/document instructions remain the client's responsibility.

## Keys, state, and metrics

```bash
leatengine keys add chat-app
leatengine keys list
leatengine keys remove chat-app
leatengine serve MODEL.gguf --host 0.0.0.0 --keys
```

Non-loopback CLI binds require a key file. `--keys` without a path selects the default file;
`--keys /path/to/keys.json` selects an explicit file. Key commands accept `--file`.
Send keys through `Authorization: Bearer KEY` or `x-api-key: KEY`. New keys start with
`leatengine-`; only their SHA-256 hashes are stored, in an owner-readable file. Changes take
effect without a server restart. Keys and identifiers partition prefix reuse.

Defaults live in `$XDG_DATA_HOME/leatengine`, falling back to `~/.local/share/leatengine`:
`keys.json` for keys and `sampling.json` for optional sampling overrides. Existing application
state is not discovered or migrated automatically. Explicit files retain their existing formats.

Metrics describe request counts, completion outcomes, tokens, timing, queue/slot occupancy,
and loaded models. They use `leatengine_` names and include API-key names as labels, not secrets.
Browser-origin requests are rejected; a web app should call through its own backend.

## Sampling overrides

`serve --sampling` reads the default sampling file; an explicit path is also accepted.
For example:

```json
{"*": {"temperature": 0.7}, "model-file-stem": {"top_p": 0.9}}
```

Request values override operator settings, which override model recommendations from GGUF
metadata and known model families. The file is reread as it changes. Malformed overrides
produce an error rather than silently falling back.
