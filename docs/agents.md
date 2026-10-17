# Agents

Coding agents and agent frameworks use `leatengine serve` as they would OpenAI's API, chat
completions with tools and streaming, the Responses API, and structured output, or Anthropic's
Messages API. Each client below was run against `serve` with Qwen3 8B on a small task, editing a
file to make a failing test pass, or a tool call and a structured answer; the versions are those
run.

## Serving

```bash
leatengine serve /path/to/model.gguf --prefix-store 16
```

- **The model's name** is the file's, without `.gguf`, as `GET /v1/models` lists it. Clients
  below take it as their model, for `Qwen_Qwen3-8B-Q4_K_M.gguf` `Qwen_Qwen3-8B-Q4_K_M`.
- **Long sessions:** the context is the model's own, as far as the slots' KV cache fits beside
  the weights; `--cache-type q8_0` holds the cache in about half the memory, so twice the tokens.
  `serve` says what it chose.
- **Many sessions:** each of the 4 slots keeps its last prompt, so a session's next turn
  prefills only what it adds. `--prefix-store GIB` keeps the prefixes of sessions whose slots
  others took, so more sessions than slots each go on from their last turn
  ([sizing](api.md#prefix-reuse-and-the-prefix-store)).
- **Reasoning models** think before each reply. A client that sends no `max_tokens` lets a reply
  run to the end of the context, as a model that loops in its reasoning may.

## Clients

### OpenCode (2.0.3)

A provider in `opencode.json`, through the AI SDK's OpenAI-compatible package:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "leatengine": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LeatEngine",
      "options": {"baseURL": "http://127.0.0.1:8080/v1", "apiKey": "unused"},
      "models": {"Qwen_Qwen3-8B-Q4_K_M": {"name": "Qwen3 8B"}}
    }
  },
  "model": "leatengine/Qwen_Qwen3-8B-Q4_K_M"
}
```

Give the model's `limit.context` as `serve` chose it, so that OpenCode compacts the conversation
before it outgrows the context.

### Codex CLI (0.154)

A model provider in `config.toml` (`$CODEX_HOME`, by default `~/.codex`), of the Responses API:

```toml
model = "Qwen_Qwen3-8B-Q4_K_M"
model_provider = "leatengine"

[model_providers.leatengine]
name = "LeatEngine"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
```

Codex sends OpenAI's hosted `web_search` and a namespace of tools for its sub-agents beside its
function tools; `serve` leaves those out, so the model is offered its function tools alone.

### Aider

Through LiteLLM's OpenAI provider:

```bash
aider --model openai/Qwen_Qwen3-8B-Q4_K_M --openai-api-base http://127.0.0.1:8080/v1 \
  --openai-api-key unused
```

### Continue CLI (1.5.47)

A model in `config.yaml`, given to `cn --config`:

```yaml
name: leatengine
version: 1.0.0
schema: v1
models:
  - name: Qwen3 8B
    provider: openai
    model: Qwen_Qwen3-8B-Q4_K_M
    apiBase: http://127.0.0.1:8080/v1
    apiKey: unused
    roles: [chat, edit, apply]
    capabilities: [tool_use]
```

### OpenAI Agents SDK (0.24)

Either model class, over chat completions or the Responses API, with tracing off, which would send
traces to OpenAI:

```python
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled

set_tracing_disabled(True)
client = AsyncOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
model = OpenAIChatCompletionsModel("Qwen_Qwen3-8B-Q4_K_M", client)
agent = Agent(name="weather", tools=[get_weather], output_type=Weather, model=model)
```

An agent's `output_type` beside its tools asks for a JSON `response_format` beside `tools`: the
reply calls a tool where the model would, and else answers in the format
([response_format](api.md#chat-completions)).

### LangChain (langchain-openai 1.7)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1", api_key="unused", model="Qwen_Qwen3-8B-Q4_K_M"
)
```

`with_structured_output` by each method, `json_schema`, `function_calling` (a `tool_choice` naming
its function) and `json_mode`, and `bind_tools` work.

### Claude Code (2.1.282)

Through Anthropic's Messages API ([`/v1/messages`](api.md#messages)), by its environment:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=unused
export ANTHROPIC_MODEL=Qwen_Qwen3-8B-Q4_K_M ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen_Qwen3-8B-Q4_K_M
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768 CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

- **The model:** `serve` answers with the loaded model whatever model a request names.
  `ANTHROPIC_MODEL` tells Claude Code, and through its system prompt the model, which model it
  is; `ANTHROPIC_DEFAULT_HAIKU_MODEL` does so for its background requests.
- **The context:** Claude Code's system prompt and its 20 tools take about 15,000 tokens before
  the task's first word. It knows no context window of a model not Anthropic's: give it the one
  `serve` chose, and an output limit that leaves its prompt room, or it compacts the conversation
  over and over. A model of 8B on 24 GB takes its whole 32,768 tokens with `--cache-type q8_0`.
- **The cache:** each turn goes on from the last, so `serve` prefills only what the turn adds; a
  user's sessions share the prefix of the system prompt and tools.

## What a small model gets wrong

The server answers each request; whether the task gets done is the model's. Qwen3 8B fixed the task
through OpenCode, Aider, Continue and Claude Code, but through Codex called a patch tool it was not
given and then edited the test rather than the code. Agents are run with larger models, as a 128 GB
machine holds.
