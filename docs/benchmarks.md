# Benchmarks

## Measure this checkout

```bash
uv run leatengine bench MODEL.gguf -p 512 -n 128 -r 3 --json
uv run leatengine bench MODEL.gguf -s 4 --json
uv run leatengine bench MODEL.gguf --chat --draft DRAFTER.gguf --json
uv run leatengine perplexity MODEL.gguf --text wiki.test.raw --ctx 512 --json
DEV=AMD uv run python scripts/validate_hardware.py /models --llama-cpp /path/to/llama.cpp/build/bin
```

Use `--recommended` with `bench --chat` to measure recommended sampling rather than greedy
replies. `perplexity --kl-base FILE` compares with a logits file produced by llama.cpp.
The validation script runs GPU tests, benchmarks, draft comparisons, and an HTTP streaming
smoke check. `--quick` reduces repetitions and omits batched/draft measurements; `--no-tests`
skips GPU tests. Reports and temporary server logs are written under ignored `reports/`.

Record the engine commit, tinygrad commit, model file/quantization, driver/compiler, hardware,
context, prompt and output lengths, sampling, sequence count, and warmup conditions. Compare
like-for-like runs and retain raw results. The script records failures in its report; inspect
those results rather than treating its process exit status as a qualification verdict.
Its built-in batched comparison uses different prompt lengths for the two engines, so it is
exploratory and must not be presented as a controlled performance comparison.

## Measured on this checkout: new architectures on the RTX 3090

Measured on an RTX 3090 (driver 610.43.03, CUDA 13.3, Linux 7.1) against llama.cpp CUDA builds
(`2e88c49c9`, and `1b70350` for Muse Glimmer, which the older one lacks), one sequence, Q4_K_M
files from Hugging Face. Speeds are
`leatengine bench -p 512 -n 128 -r 2` against `llama-bench -p 512 -n 128 -r 2`, in tokens per
second. Quality is `leatengine perplexity --kl-base` against llama.cpp's logits of the first 8
chunks of 512 tokens of wikitext-2's test set: mean KL divergence and top-token agreement. For
scale, the last column gives llama.cpp's own CPU backend against its CUDA one, on 2 chunks.

| | llama.cpp pp512 | leatengine pp512 | llama.cpp tg128 | leatengine tg128 | KL, top-1 | llama.cpp CPU vs CUDA |
|---|---:|---:|---:|---:|---:|---:|
| Mellum 2.1 12B A2.5B Thinking | 7790 | 8126 | 294 | 311 | 0.0078, 95.8% | 0.0095 |
| LFM2.5 24B A2B | 5998 | 5135 | 280 | 292 | 0.051, 89.9% | 0.039, 90.7% |
| GLM-4.7-Flash, 7 chunks | 3733 | 1088 | 151 | 98 | 0.0076, 96.1% | 0.0088 |
| Muse Glimmer 30B | 1631 | 1384 | 45.1 | 46.9 | 0.0021, 98.9% | |

LFM2.5's short convolutions multiply three outputs of one projection, which makes it unusually
sensitive to rounding on every backend; its prefill trails llama.cpp in the int8 matrix kernels'
handling of its small per-expert batches. GLM-4.7-Flash's 8th chunk scores a perplexity in the
thousands on every backend, where they diverge, so its row covers the first 7; its multi-head
latent attention takes plain tinygrad ops for each head's projections into and out of the
compressed dimensions, which its speed shows.

## Inherited RTX 3090 results

The table below is historical evidence copied from the original
[Leat README at eb5682ac62134425d2ac7d8add6c8b466f12a18b](https://github.com/alkinun/leat/blob/eb5682ac62134425d2ac7d8add6c8b466f12a18b/README.md#measurements).
It was not rerun as part of the extraction. Columns labelled `leat` refer to that original
implementation. These are NVIDIA results, not Strix Halo results, and the original source
does not record a complete machine configuration for reproducing every row.

RTX 3090, one sequence, in tokens per second; Q4_K_M files but for gpt-oss's, MXFP4. llama.cpp is b11372 with CUDA, and b9691 for Qwen3.6. leat's numbers include sampling on the device and reading the token back: after the prompt for pp512, after every token for tg128.

| | llama.cpp pp512 | leat pp512 | llama.cpp tg128 | leat tg128 |
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
