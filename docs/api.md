# HTTP API and configuration

Start with `leatengine serve MODEL.gguf`. The default address is `http://127.0.0.1:8080`.
The API implements a subset of OpenAI chat and text completions, responses, and embeddings, and
of Anthropic's Messages, plus model-management, metrics and health extensions. It serves no HTML,
frontend, or browser application.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /v1/models` | List configured chat models and their load status |
| `POST /v1/models/load` | Load a configured model, replacing the current chat model |
| `POST /v1/chat/completions` | Generate one reply, optionally streamed |
| `POST /v1/completions` | Continue a prompt, or fill in the middle, optionally streamed |
| `POST /v1/responses` | Generate one reply as a response, statelessly, optionally streamed |
| `POST /v1/messages` | Generate one reply as Anthropic's Messages API does, optionally streamed |
| `POST /v1/messages/count_tokens` | Count a Messages request's prompt tokens |
| `POST /v1/embeddings` | Embed text using the separately configured embedding model |
| `GET /metrics` | Prometheus metrics with the `leatengine_` prefix |
| `GET /health` | `200` once a chat model is loaded and warmed up, `503` until then |

All routes but `/health` use the same authentication when keys are enabled; `/health` needs no
key, so that a supervisor can wait for the server to be ready, as llama.cpp's server allows.
Request bodies are limited to 64 MiB. Errors use an `error` object with `message`, `type`,
`param`, and `code`, as OpenAI's API does, or of `/v1/messages`, as Anthropic's does. A generation
error after streaming starts arrives as an SSE error event; clients must handle that as well as
HTTP errors.

## Models

`serve` accepts files, directories containing GGUFs, or models `leatengine pull` downloaded, by
their `OWNER/NAME:QUANT`. Model IDs are filename stems. The first
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
switch models: generation uses the currently loaded model. Use the explicit load endpoint. While
no model is loaded, as after a load failed, completions are answered 503, as `/health` is.

## Chat completions

An OpenAI Python client can use `base_url="http://127.0.0.1:8080/v1"` and an API key (any
placeholder when authentication is disabled). The client library is not a runtime dependency.

Supported controls include `messages`, `stream`, `stream_options.include_usage`, `max_tokens`
or `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `logit_bias`, `seed`, `stop`, `tools`, `tool_choice`,
`response_format`, `logprobs`, and `top_logprobs`. Templates determine which message roles and
tool formats a model understands.

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
- `tool_choice` is `auto`, `none`, `required`, or a function to call, as
  `{"type": "function", "function": {"name": "weather"}}`, of which the template is then told
  alone. A reply of `required` or of a function begins with a call, after any reasoning: the
  marker of a call of the model's format, as Qwen's `<tool_call>`, then the call as the model
  writes it, or where its vocab has no such marker, a JSON call as Llama 3's, of a tool's name.
  gpt-oss's and Muse Glimmer's calls come in their formats' channels, which no reply is held to:
  of them, `required` and a function are refused.
- `response_format` of type `json_object` holds the reply's answer to a JSON object, and of
  type `json_schema` to its schema's JSON, `strict` or not: only tokens that can continue it are
  sampled, and the reply ends, `finish_reason: "stop"`, once the JSON is whole. Schemas may use
  `type`, `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`,
  `enum`, `const`, `anyOf`, `oneOf` (taken as `anyOf`), and `$ref` to `$defs`, recursive too;
  other keywords, as `pattern`, `format` or `minimum`, are refused with an error rather than
  ignored. Properties come in the schema's order. A reasoning model reasons first, freely, as its
  template opens or allows it. A reply cut off by `max_tokens` holds JSON cut short. Beside
  `tools` it may call, as agent SDKs ask for structured output, a reply may call one in place of
  the JSON: after the marker of a call of the model's format, as Qwen's `<tool_call>`, or as a
  JSON object of a tool's `name` and its `parameters` or `arguments`, as Llama 3 calls; harmony's
  and ATEM's calls come before the answer, as they would. A reply begins a call only where the
  model's own likeliest token would, and else answers in the format, as it would have answered
  without one, rather than call for want of the words it meant to answer in.
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
- Replies include usage and a `timings` extension, as llama.cpp's server gives it, with
  `tokenize_ms`, the time the prompt took to tokenize: of a prompt that starts as one of the
  last 8 did, as an agent's turn starts as its last, only what follows their last control token in
  common is tokenized. Cached token counts appear in `usage.prompt_tokens_details.cached_tokens`;
  streamed usage requires `include_usage`.

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
and `truncation` are ignored. Tools of other kinds, such as OpenAI's hosted web search or a
`namespace` of tools, are left out, the model not offered them, so that clients that send them by
default, as Codex CLI does, still work.

A reply's output is a `reasoning` item of its reasoning's text, a `message` of its text, and a
`function_call` per call. A reply cut off by `max_output_tokens` or the context is `incomplete`.
`usage` counts no reasoning tokens apart. Streams send the Responses API's events: each output
item added, its text's deltas, and it done, then `response.completed` or
`response.incomplete` with the whole response; calls come once the reply is done.

## Messages

`POST /v1/messages` answers as Anthropic's Messages API does, for clients such as Claude Code and
Anthropic's SDKs, which send their key as `x-api-key` or as a bearer token. `system`, as text or
text blocks, and `messages` of text, `image`, `tool_use`, `tool_result` and `thinking` blocks become
the chat completion's messages: a tool result goes back as a tool's message, and thinking as the
model's reasoning before its calls. A `system` message among them, as Claude Code says its
environment after the user's first words, stays where it is, for the model's template to show.
Client `tools`, `tool_choice` (`auto`, `any`, `tool`, `none`), `max_tokens`, `stop_sequences` and
`stream` work as for chat completions, and so do `temperature`, `top_p` and `top_k`, which
Anthropic's API no longer takes but older clients send. `output_config.format`, a JSON schema, holds
the reply to it as `response_format` does. `output_config.effort` is the reasoning effort, `max` the
highest the model takes; `thinking` of type `disabled` asks for none, `enabled` without an effort
for medium, and `adaptive` or `between_tools` leaves it to the model, where its template takes an
effort at all. `metadata.user_id` keeps cached prefixes to that user's prompts, as `user` does; of
Claude Code's, its device's and account's, so that a user's sessions share their prefix. Anthropic's
server tools, such as web search, are left out, the model not offered them; `cache_control` is
ignored, as the server reuses prefixes by itself. The request's `model` is ignored: the loaded model
answers.

A reply's content is a `thinking` block of its reasoning, of an empty `signature`, a `text` block,
and a `tool_use` block per call; `stop_reason` is `tool_use`, `max_tokens`, `stop_sequence` or
`end_turn`. `usage.input_tokens` counts the prompt's tokens past those read from the cache,
`cache_read_input_tokens`. Streams send Anthropic's events: `message_start`, each content block's
`content_block_start`, deltas and `content_block_stop`, then `message_delta` with the stop reason
and usage, and `message_stop`; tool uses come once the reply is done.
`POST /v1/messages/count_tokens` gives a request's `input_tokens` as the loaded model's template
renders it.

## Embeddings

Start with `--embed /path/to/embedding-model.gguf`. The endpoint accepts `input` as a string
or a nonempty list of up to 256 strings and returns float vectors, indexes, model ID, and token
usage. It uses the configured embedding model, independently of the chat model. Without a
configured embedding model, the endpoint returns 404. Inputs are truncated to the embedding
context; model-specific query/document instructions remain the client's responsibility.

## Prefix reuse and the prefix store

Each of the server's slots, 4 by default, keeps the keys and values of the last prompt it ran, and
a prompt that starts as a slot's tokens do, of the same key and user, prefills only the rest, as a
conversation's next turn does. With more conversations than slots, as several agents' sessions,
each new one takes a slot from another and that one's next turn prefills from the start.

`serve --prefix-store GIB` keeps, in that much host memory, what a slot held when another prompt
took it, if 256 tokens or more, letting the least recently used go first; a prompt that goes on
from one of them copies it back into a slot rather than prefilling it again. Such tokens count as
cached in `usage.prompt_tokens_details.cached_tokens`, and `/metrics` counts them as
`leatengine_stored_tokens_total` too, with the store's size as `leatengine_prefix_store_bytes`.

Size the store by the KV cache a token takes, about what `serve` says the slots' cache takes divided
by their context: 128 KiB in f16 for Llama 3.1 8B, half that with `--cache-type q8_0`, so that 8
sessions of 30,000 tokens take about 30 GiB. Copying a prefix back takes far less than prefilling
it; copying one to the store, as its slot is taken, takes several times as long as copying it back
on a GPU of its own memory. On an APU, as Strix Halo's, the GPU's memory and the host's are the
same, so the store takes from what the models and their caches have.

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
