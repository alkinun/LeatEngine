<img src="docs/logo.svg" alt="LeatEngine logo" width="48">

# LeatEngine

A minimal GGUF inference engine built on [tinygrad](https://github.com/tinygrad/tinygrad),
with a terminal CLI and an OpenAI-compatible HTTP API. Strix Halo is the primary hardware
target; NVIDIA kernels and CPU reference execution are also retained.

**Early development:** physical Strix Halo validation is pending. AMD kernels are tested on
tinygrad's RDNA 3 emulator, and the engine is measured against llama.cpp on an RTX 3090. See
[hardware status](docs/strix-halo.md) and [benchmarks](docs/benchmarks.md).

## Install

Linux, Python 3.12+, Git, and [uv](https://docs.astral.sh/uv/) are required for these commands.
GPU execution also requires the backend's driver and compiler: NVIDIA's open kernel driver
and NVRTC, or AMD's KFD device and clang with its AMDGPU target, as most distributions ship it.
ROCm is not needed. See the [Strix Halo guide](docs/strix-halo.md).

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

`run` and `serve` compile a model's graphs before its first reply. tinygrad keeps the compiled
kernels in `~/.cache/tinygrad`, so a model starts faster after its first time: on an RTX 3090,
Llama 3.1 8B's warm-up takes about 20 seconds the first time and 13 after, and with `serve`'s 4
slots, 26 and 18.

`DEV=AMD`, `DEV=NV`, or `DEV=CPU` explicitly selects a backend. Without `DEV`, the CLI checks
for NVIDIA and then AMD devices before leaving selection to tinygrad. CPU execution is useful
for reference tests and small models; optimized inference uses the GPU kernels.

To serve other machines, create a key and enable authentication:

```bash
leatengine keys add chat-app
leatengine serve /path/to/model.gguf --host 0.0.0.0 --keys
```

The key is shown once. Clients send it as `Authorization: Bearer KEY`. The server provides HTTP;
use an appropriate TLS proxy when encrypted transport is needed. See
[API and configuration](docs/api.md).

## Supported models

LeatEngine runs GGUF files by their `general.architecture`. Support depends on the architecture
and tensors in the file, not on the model's name.

| GGUF architecture | Example models |
|---|---|
| `llama` | Llama 3.x, Mistral 7B, Mistral Small 3 (with vision) |
| `qwen2`, `qwen3`, `qwen3moe` | Qwen2.5, Qwen3, Qwen3 30B A3B, Qwen3-Embedding |
| `qwen35`, `qwen35moe` | Qwen3.5 and Qwen3.6, with vision and their MTP drafters |
| `gemma3`, `gemma4` | Gemma 3 and Gemma 4, with vision and Gemma 4's assistant drafter |
| `gemma-embedding2` | EmbeddingGemma 2 |
| `gpt-oss` | gpt-oss 20B |
| `phi3` | Phi-3, Phi-4-mini |
| `mellum` | Mellum 2.1 12B A2.5B |
| `muse-glimmer` | Muse Glimmer 30B (text only) |
| `lfm2moe` | LFM2.5 24B A2B |
| `deepseek2` | GLM-4.7-Flash (multi-head latent attention) |

The GPU kernels take Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and MXFP4 weights for prefill and decode, and Q4_1 and Q5_1 for
decode. Other weights, F32, F16 and BF16, and the prefill of Q4_1 and Q5_1, run on tinygrad's
reference ops, which are slower.

## Capabilities

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
