<img src="docs/logo.svg" alt="LeatEngine logo" width="48">

# LeatEngine

A minimal GGUF inference engine built on [tinygrad](https://github.com/tinygrad/tinygrad),
with a terminal CLI and an OpenAI-compatible HTTP API. Strix Halo is the primary hardware
target; NVIDIA kernels and CPU reference execution are also retained.

**Early development:** physical Strix Halo validation is pending. AMD kernels have been tested
on tinygrad's RDNA 3 emulator. Historical RTX 3090 results come from the original project,
not a new hardware qualification of this repository. See [hardware status](docs/strix-halo.md)
and [benchmarks](docs/benchmarks.md).

## Install

Linux, Python 3.12+, Git, and [uv](https://docs.astral.sh/uv/) are required for these commands.
GPU execution also requires the backend's driver and compiler: NVIDIA's open kernel driver
and NVRTC, or AMD's KFD device and ROCm comgr. See the [Strix Halo guide](docs/strix-halo.md).

```bash
uv tool install --python 3.14 'git+https://github.com/alkinun/LeatEngine.git'
leatengine --help
```

For development or to build a wheel:

```bash
git clone https://github.com/alkinun/LeatEngine.git
cd LeatEngine
uv sync --locked --python 3.14
uv run leatengine --help
uv build
# Install the wheel in a separate tool environment:
uv tool install --force dist/leatengine-0.0.1-py3-none-any.whl
```

The wheel resolves dependencies, including tinygrad at an exact Git commit; it is not an
offline bundle. Model weights are supplied separately as local GGUF files.

## Run

```bash
leatengine run /path/to/model.gguf
leatengine serve /path/to/model.gguf
```

The server listens on `127.0.0.1:8080`, with 4 slots and a context of 4096 tokens by default.
Use `--max-context` and `--slots` to fit available memory. `python -m leatengine` exposes the
same commands in the environment where the package is installed.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":128,"stream":true}'
```

`DEV=AMD`, `DEV=NV`, or `DEV=CPU` explicitly selects a backend. Without `DEV`, the CLI checks
for NVIDIA and then AMD devices before leaving selection to tinygrad. CPU execution is useful
for reference tests and small models; optimized inference uses the GPU kernels.

To serve other machines, create a key and enable authentication:

```bash
leatengine keys add chat-app
leatengine serve /path/to/model.gguf --host 0.0.0.0 --keys
```

The key is shown once. Clients send it as `Authorization: Bearer KEY`. The server provides HTTP;
use an appropriate TLS proxy when encrypted transport is needed. See [API and configuration](docs/api.md).

## Capabilities

- GGUF models and quantized weights; supported architecture families include Llama/Mistral,
  Qwen2, Qwen3, Qwen3 MoE, Qwen3.5 dense/MoE, Gemma3, Gemma4, Gemma Embedding2, gpt-oss, Phi3,
  Mellum, Muse Glimmer (text), LFM2 MoE, and DeepSeek V2-style multi-head latent attention as
  GLM-4.7-Flash uses it.
  Compatibility depends on the architecture and tensors in the file, not just its model name.
- Continuous batching, chunked prefill, prefix reuse, and sliding-window KV caches.
- Speculative decoding with supported MTP layers or assistant drafters.
- Vision through supported `mmproj` GGUFs; embeddings through `serve --embed MODEL.gguf`.
- Streaming chat, reasoning output, and tool-call formatting. Applications execute tools.
- API keys, per-client cache isolation, and Prometheus metrics.

| Command | Purpose |
|---|---|
| `run` | Chat in the terminal; `/image PATH` attaches an image when supported |
| `serve` | Serve local GGUF files or directories through HTTP |
| `keys` | Add, list, and remove API keys |
| `bench` | Measure prefill and decode throughput |
| `perplexity` | Measure perplexity or compare logits with llama.cpp |

Use `leatengine COMMAND --help` for arguments. The server keeps one chat model loaded at a time,
with an optional embedding model alongside it. There is no frontend or application service.

## Development

[Development and tests](docs/development.md) · [API](docs/api.md) ·
[Benchmarks](docs/benchmarks.md) · [Strix Halo](docs/strix-halo.md)

MIT licensed, copyright 2026 alkinun. Extracted with fresh history from
[alkinun/leat at eb5682ac62134425d2ac7d8add6c8b466f12a18b](https://github.com/alkinun/leat/tree/eb5682ac62134425d2ac7d8add6c8b466f12a18b).
The inference implementation and its tests retain that source's attribution. This repository
is independent of the original app and appliance; clients integrate through HTTP.
