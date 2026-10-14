# Benchmarks

This page has three parts: results measured on this checkout on an RTX 3090, historical results
from the original Leat project, and how to measure a checkout yourself. Physical Strix Halo
results are still pending; see the [Strix Halo guide](strix-halo.md).

## RTX 3090 results

Measured with the engine at commit `d17c2d2` on an RTX 3090 (24 GB; driver 610.43.03, CUDA 13.3,
Linux 7.1, Python 3.14), one sequence, Q4_K_M GGUFs from Hugging Face. Every row compares against
the same llama.cpp CUDA build, commit `1b70350`, the first tested build that runs Muse Glimmer.

### Speed

Tokens per second: prefill of a 512-token prompt (pp512) and decode of 128 tokens (tg128), from
`leatengine bench MODEL -p 512 -n 128 -r 3` and `llama-bench -m MODEL -p 512 -n 128 -r 3`.
LeatEngine's numbers include sampling on the GPU and reading each token back to the host.

| Model | Architecture | llama.cpp pp512 | LeatEngine pp512 | llama.cpp tg128 | LeatEngine tg128 |
|---|---|---:|---:|---:|---:|
| Llama 3.1 8B Instruct | llama | 5369 | 4784 | 147.2 | 150.1 |
| Qwen3 8B | qwen3 | 5098 | 4671 | 141.3 | 146.7 |
| Mellum 2.1 12B A2.5B Thinking | mellum | 8070 | 8349 | 291.6 | 312.6 |
| LFM2.5 24B A2B | lfm2moe | 6013 | 5238 | 283.7 | 293.6 |
| GLM-4.7-Flash (30B A3B) | deepseek2 | 3756 | 2455 | 150.9 | 138.4 |
| Muse Glimmer 30B | muse-glimmer | 1626 | 1392 | 45.1 | 47.0 |

### Agreement with llama.cpp

`llama-perplexity` saves llama.cpp's logits for the first 8 chunks of 512 tokens of wikitext-2's
test set (`--kl-divergence-base`), and `leatengine perplexity --kl-base` scores the same tokens.
The table gives both engines' perplexity, and LeatEngine's mean KL divergence from llama.cpp and
how often both pick the same most likely token. The prompt path runs each chunk as a prompt; the
decode path runs the scored half one token at a time, as generation does. Lower KL is closer.

| Model | llama.cpp perplexity | LeatEngine perplexity | KL, top-1: prompt path | KL, top-1: decode path |
|---|---:|---:|---:|---:|
| Llama 3.1 8B Instruct | 7.994 | 7.983 | 0.0011, 98.3% | 0.0011, 98.1% |
| Qwen3 8B | 10.214 | 10.171 | 0.0028, 97.2% | 0.0027, 97.3% |
| Mellum 2.1 12B A2.5B Thinking | 11.825 | 11.801 | 0.0080, 95.9% | 0.0082, 95.9% |
| LFM2.5 24B A2B | 16.494 | 16.592 | 0.057, 88.6% | 0.056, 89.5% |
| GLM-4.7-Flash, first 7 chunks | 7.153 | 6.896 | 0.032, 95.0% | 0.040, 95.0% |
| Muse Glimmer 30B | 6.585 | 6.597 | 0.0021, 98.9% | 0.0021, 98.4% |

Both engines round differently, so some divergence is expected; how much depends on the model.
For scale, llama.cpp also diverges from itself:

- **LFM2.5** is unusually sensitive to rounding: its short convolutions multiply three outputs of
  one projection. llama.cpp's own CPU backend scores a KL of 0.039 (90.7% top-1) against its
  CUDA backend on these chunks.
- **GLM-4.7-Flash** is compared on the first 7 chunks: on the 8th the model reaches a perplexity
  in the thousands on every backend, and the backends diverge widely there. llama.cpp `1b70350`
  and the older `2e88c49c9` differ from each other by a KL of 0.039 on the 7 chunks (perplexity
  7.15 and 6.95). LeatEngine scores 0.011 against `2e88c49c9`, and its layers match an independent
  float64 implementation of the model.

## Historical results from the original Leat

These measurements come from the original
[Leat README at eb5682ac62134425d2ac7d8add6c8b466f12a18b](https://github.com/alkinun/leat/blob/eb5682ac62134425d2ac7d8add6c8b466f12a18b/README.md#measurements)
and were not rerun for this repository: the `leat` columns are that implementation, not this
checkout. They are NVIDIA results, and the source does not record the full machine
configuration. Llama 3.1 8B and Qwen3 8B also appear above, measured on this checkout.

RTX 3090, one sequence, tokens per second. Q4_K_M files except gpt-oss, which is MXFP4.
llama.cpp is b11372 with CUDA (b9691 for Qwen3.6).

| Model | llama.cpp pp512 | leat pp512 | llama.cpp tg128 | leat tg128 |
|---|---:|---:|---:|---:|
| Llama 3.2 3B Instruct | 10907 | 10185 | 274.4 | 283.1 |
| Llama 3.1 8B Instruct | 5422 | 4829 | 147.7 | 152.9 |
| Mistral 7B Instruct v0.3 | 5454 | 4883 | 156.8 | 161.9 |
| Mistral Small 3.2 24B Instruct | 1996 | 1636 | 54.9 | 58.0 |
| Qwen2.5 7B Instruct | 5830 | 5263 | 152.6 | 165.7 |
| Qwen3 8B | 5296 | 4745 | 142.0 | 149.2 |
| Qwen3 30B A3B | 4697 | 5036 | 213.1 | 233.1 |
| Qwen3.6 35B A3B | 3516 | 3708 | 168.0 | 192.2 |
| Qwen3.8 27B | 1434 | 1188 | 39.4 | 42.4 |
| Gemma 3 4B it | 9795 | 8436 | 203.0 | 226.8 |
| Gemma 3 12B it | 3463 | 2749 | 89.0 | 97.6 |
| Gemma 4 26B A4B it | 4823 | 4919 | 158.2 | 190.6 |
| Phi-4-mini Instruct | 10382 | 8593 | 237.4 | 239.6 |
| gpt-oss 20B | 6101 | 5846 | 213.2 | 230.0 |

## Measuring a checkout

```bash
uv run leatengine bench MODEL.gguf -p 512 -n 128 -r 3 --json
uv run leatengine bench MODEL.gguf -s 4 --json
uv run leatengine bench MODEL.gguf --chat --draft DRAFTER.gguf --json
uv run leatengine perplexity MODEL.gguf --text wiki.test.raw --ctx 512 --json
uv run leatengine perplexity MODEL.gguf --kl-base BASE.kld --json
DEV=AMD uv run python scripts/validate_hardware.py /models --llama-cpp /path/to/llama.cpp/build/bin
```

- `bench -s N` decodes N sequences at once; `bench --chat` decodes replies to chat prompts,
  greedily or with `--recommended` sampling, as a drafter guesses them in use.
- `perplexity --kl-base` compares with a file written by `llama-perplexity --kl-divergence-base`;
  add `--decode` for the decode path and `--chunks N` to score fewer chunks.
- `scripts/validate_hardware.py` runs the GPU tests, speed against llama.cpp, draft comparisons
  and a streamed HTTP request, and writes a report under the ignored `reports/` directory.
  `--quick` runs fewer repetitions and skips the batched and draft runs; `--no-tests` skips the
  GPU tests. Its process exit status is not a verdict: read the failures in its report. Its
  batched comparison uses different prompt lengths for the two engines, so treat it as
  exploratory rather than a controlled comparison.

Record the engine and tinygrad commits, the llama.cpp commit and build options, the model file
and quantization, the hardware, driver and compiler, the context, prompt and output lengths,
sampling, the number of sequences, and warmup. Compare like with like and keep the raw results.
