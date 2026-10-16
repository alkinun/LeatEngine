# HTTP API and configuration

Start with `leatengine serve MODEL.gguf`. The default address is `http://127.0.0.1:8080`.
The API implements a subset of OpenAI chat and text completions, responses, and embeddings, plus
model-management, metrics and health extensions. It serves no HTML, frontend, or browser
application.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /v1/models` | List configured chat models and their load status |
| `POST /v1/models/load` | Load a configured model, replacing the current chat model |
| `POST /v1/chat/completions` | Generate one reply, optionally streamed |
| `POST /v1/completions` | Continue a prompt, or fill in the middle, optionally streamed |
| `POST /v1/responses` | Generate one reply as a response, statelessly, optionally streamed |
| `POST /v1/embeddings` | Embed text using the separately configured embedding model |
| `GET /metrics` | Prometheus metrics with the `leatengine_` prefix |
| `GET /health` | `200` once a chat model is loaded and warmed up, `503` until then |

All routes but `/health` use the same authentication when keys are enabled; `/health` needs no
key, so that a supervisor can wait for the server to be ready, as llama.cpp's server allows.
Request bodies are limited to 64 MiB. Errors use an `error` object with `message`, `type`,
`param`, and `code`. A generation error after streaming starts arrives as an SSE error event;
clients must handle that as well as HTTP errors.

## Models

`serve` accepts files or directories containing GGUFs. Model IDs are filename stems. The first
chat model loads at startup. Recognized projectors and MTP files are excluded from the list. A
GGUF split in parts, as llama.cpp's `gguf-split` writes `NAME-00001-of-00003.gguf` and on, is
given by its first part, its ID `NAME`; a directory's later parts are not models of their own.

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
`frequency_penalty`, `repetition_penalty`, `logit_bias`, `seed`, `stop`, `tools`, `tool_choice`
(`auto` or `none`), `response_format`, `logprobs`, and `top_logprobs`. Templates determine which
message roles and tool formats a model understands.

- Penalties lower text tokens alone, never a format's markup, such as `<tool_call>`, which a
  reply must repeat. `repetition_penalty`, as vLLM's and Hugging Face's, divides a positive logit
  and multiplies a negative one, of each token of the prompt or of the reply so far; then
  `presence_penalty` comes off each token of the reply so far, and `frequency_penalty` as many
  times as it occurs, as OpenAI's. llama.cpp's server counts all three over the last 64 tokens
  of prompt and reply instead. `logit_bias` adds to the logits of the token IDs it names, -100 to
  100 each, before the penalties.
- Streaming uses Server-Sent Events: JSON `data:` records followed by `data: [DONE]` on success.
  Disconnecting cancels the completion and releases its slot.
- Reasoning is returned as `reasoning_content`. `reasoning_effort` and
  `chat_template_kwargs` control supported templates; explicit template options take precedence.
- Tool calls are returned in OpenAI-style `tool_calls`. The engine does not execute them. It
  reads each model's own call format: Llama 3's JSON replies, Qwen's `<tool_call>` blocks of JSON
  or Qwen3.5's XML, GLM-4.7's argument tags, Gemma 4's calls, LFM2's Python-style call lists,
  gpt-oss's harmony messages, and Muse Glimmer's ATEM messages. Arguments written as text are
  typed by the tool's JSON schema. A reply cut off by `max_tokens` or the context keeps
  `finish_reason: "length"` even when it holds complete calls, as a later call may be missing.
- `response_format` of type `json_object` holds the reply's answer to a JSON object, and of
  type `json_schema` to its schema's JSON, `strict` or not: only tokens that can continue it are
  sampled, and the reply ends, `finish_reason: "stop"`, once the JSON is whole. Schemas may use
  `type`, `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`,
  `enum`, `const`, `anyOf`, `oneOf` (taken as `anyOf`), and `$ref` to `$defs`, recursive too;
  other keywords, as `pattern`, `format` or `minimum`, are refused with an error rather than
  ignored. Properties come in the schema's order. A reasoning model reasons first, freely, as its
  template opens or allows it. A reply cut off by `max_tokens` holds JSON cut short. A request
  may not yet give both `tools` and a JSON `response_format`.
- `logprobs: true` gives each token's log-probability, and `top_logprobs` those of up to 20 of
  the likeliest tokens, in `choices[].logprobs.content`, streamed with the chunks: of the model's
  distribution as its logits give it, before temperature, the other sampling controls and a
  response format, as llama.cpp's and vLLM's servers give them by default. Every generated token
  has one, the reasoning's too, but an end of generation. A completion that asks for them never
  decodes speculatively, and each of its steps runs one more small graph, for the
  log-probabilities alone.
- Image content parts accept `image_url` with base64 `data:` URLs. Remote image URLs are not
  fetched. Supply `--mmproj` or a recognized projector beside the model.
- `service_tier: "flex"` schedules a completion behind foreground requests.
- `safety_identifier`, or `user` when absent, isolates cached prefixes within an API key.
  Clients sharing a key must supply distinct identifiers when their prompt caches should be
  separate.
- `stop` takes a string or up to 16 strings, each of 256 characters at most.
- Replies include usage and a `timings` extension. Cached token counts appear in
  `usage.prompt_tokens_details.cached_tokens`; streamed usage requires `include_usage`.

This is not a complete OpenAI implementation: multiple choices (`n` above 1) are unsupported.
Audio, file-storage, and tool-execution endpoints are not implemented. A prompt must fit the
configured context; clients manage conversation history.

## Text completions

`POST /v1/completions` continues a prompt as the model writes it, with no chat template, as
OpenAI's legacy completions and llama.cpp's `/completion` do. `prompt` is a string, whose text of
control tokens, such as `<|im_start|>`, is read as those tokens, as llama.cpp reads it; a list of
one string; or a list of token IDs. `max_tokens` defaults to 16, as OpenAI's does.

With `suffix`, the reply fills in the middle, as code-completion clients ask: the prompt and the
suffix go between the model's fill-in-the-middle tokens, `[FIM_PRE]prompt[FIM_SUF]suffix[FIM_MID]`,
in llama.cpp's default order, neither's text read as tokens. The tokens are those GGUF metadata
names, else those of the texts llama.cpp knows, such as Qwen's `<|fim_prefix|>`; a model without
them refuses a suffix. Unlike llama.cpp's `/infill`, no repository header or other files are added
and nothing is cut: the prompt must fit the context.

Sampling controls, `stop`, `seed`, `stream`, and `stream_options` work as for chat completions.
`logprobs` is a count of the likeliest tokens, 0 to 20, whose log-probabilities come with each
token's, as OpenAI's text completions give them: `tokens`, `token_logprobs`, `top_logprobs` by
token text, and `text_offset`. `logit_bias` works as for chat completions; `echo` and `best_of`
are not supported.

## Responses

`POST /v1/responses` answers as OpenAI's Responses API does, of a stateless subset: the server
keeps no response, so `previous_response_id`, `conversation`, and `background` are refused, and a
client sends the whole conversation as `input` each time, its past output items too, as clients
do with `store: false`. `input` is a string or a list of messages, `function_call` and
`function_call_output` items, and `reasoning` items, whose text goes back to the model as its
reasoning before the calls it made, as gpt-oss's template keeps it. `instructions`, function
`tools`, `tool_choice`, `text.format` (as `response_format`), `reasoning.effort`,
`max_output_tokens`, the sampling controls, and `stream` work as for chat completions; `include`
and `truncation` are ignored. Other tools, such as web search, are refused.

A reply's output is a `reasoning` item of its reasoning's text, a `message` of its text, and a
`function_call` per call. A reply cut off by `max_output_tokens` or the context is `incomplete`.
`usage` counts no reasoning tokens apart. Streams send the Responses API's events: each output
item added, its text's deltas, and it done, then `response.completed` or
`response.incomplete` with the whole response; calls come once the reply is done.

## Embeddings

Start with `--embed /path/to/embedding-model.gguf`. The endpoint accepts `input` as a string
or a nonempty list of up to 256 strings and returns float vectors, indexes, model ID, and token
usage. It uses the configured embedding model, independently of the chat model. Without a
configured embedding model, the endpoint returns 404. Inputs are truncated to the embedding
context; model-specific query/document instructions remain the client's responsibility.

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
POST requests from web pages, which carry an `Origin` header, are refused, so that another site
cannot make the server generate or load models; a web app should call through its own backend.

## Sampling overrides

`serve --sampling` reads the default sampling file; an explicit path is also accepted.
For example:

```json
{"*": {"temperature": 0.7}, "model-file-stem": {"top_p": 0.9}}
```

Request values override operator settings, which override model recommendations from GGUF
metadata and known model families. The file is reread as it changes. Malformed overrides
produce an error rather than silently falling back.
