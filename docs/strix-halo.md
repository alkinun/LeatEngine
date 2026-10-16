# Strix Halo bring-up

The initial target is a Ryzen AI Max+ 395 system with 128 GB unified memory and a Radeon
8060S (`gfx1151`, RDNA 3.5). The machine has been ordered but has not yet been tested.
This document is a qualification procedure, not a claim of measured support or performance.

## Current evidence

| Execution path | Evidence |
|---|---|
| CPU reference | Generated-model correctness tests |
| AMD | Kernel and tiny-model tests on tinygrad's RDNA 3 emulator |
| NVIDIA | GPU tests and real-model comparisons with llama.cpp on an RTX 3090 ([benchmarks](benchmarks.md)) |
| Physical Strix Halo | Pending |

The current kernels select their RDNA 3 path for architecture major 11. Whether this gives
correct, efficient execution on `gfx1151` must be confirmed on the physical machine.

One check can only run there: decode attention over heads wider than 512 dimensions, the 576 of
GLM-4.7-Flash's multi-head latent attention. Its kernel uses more than 128 VGPRs, and the
emulator misreads 16-bit operands in those registers, so the emulator skips those cases. The same
kernel passes on NVIDIA, and on the emulator when compiled within 128 VGPRs.
`DEV=AMD uv run pytest -m gpu -k attention_groups` runs them on the physical machine.

## Prepare the host

Use Linux with firmware and an amdgpu driver that support the device. Install uv, Git, and
clang with its AMDGPU target, which compiles the kernels; ROCm is not needed. Ensure the user can
access `/dev/kfd` and the relevant render device; device permissions depend on the distribution.

Record the distribution, kernel, firmware/BIOS, compiler, memory configuration, and power mode.
Inspect available GPU memory before selecting models or contexts. Firmware UMA allocation and
Linux TTM/GTT limits can affect usable memory; verify their behavior on this machine before
applying tuning. No kernel-memory settings are assumed validated here.

Weights load in buffers of up to 1 GiB of adjacent tensors, which tinygrad requests from the
driver as VRAM. On an APU whose firmware carve-out is smaller than its GTT, amdgpu serves such
requests from GTT, so a small carve-out and a large GTT, as AMD's
[Strix Halo guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
recommends, should let the GPU's buffers take most of memory. This comes from the driver's
source; confirming it on the machine is part of qualification. A load that does not fit stops
with the size and tensors that failed.

`leatengine doctor` reports the kernel, access to `/dev/kfd` and the render node, the GPU and its
firmware, the carve-out, GTT and TTM's limit, the memory free for a model's weights and KV cache, of
which a model takes its context, and clang; it compiles a kernel for the GPU, runs one there, and
says in which pool a 1 GiB buffer lands. Its report holds most of what the qualification records,
and what to change where something will not do.

```bash
uv sync --locked --python 3.14
uv run leatengine doctor
DEV=AMD uv run python -c "from tinygrad import Device; print(Device['AMD'].arch)"
DEV=AMD uv run pytest -m gpu
DEV=AMD uv run python scripts/validate_hardware.py /models --llama-cpp /path/to/llama.cpp/build/bin
```

Run the real-GPU tests in a single process. Start with a model that comfortably fits memory,
then increase model size, context, and simultaneous sequences. Test vision, embeddings, and
speculation with their matching weights before describing them as hardware-verified.

## Qualification

1. Confirm device detection, allocation, kernel correctness, and stable repeated generation.
   Load a model close to the memory configuration's limit.
2. Compare model quality against reference execution and llama.cpp using matching GGUF files.
3. Measure prefill/decode, first-token latency, and memory at documented contexts and slot counts.
4. Exercise concurrent requests, cancellation, model switching, vision, and embeddings.
5. Repeat longer runs to expose memory pressure, thermal effects, and startup/compile costs.

For llama.cpp comparisons, record the exact commit and build options for each backend, such as
Vulkan or HIP. Keep model files, quantization, prompt lengths, and generation settings comparable;
report differences rather than mixing them into one headline number.

Changing memory defaults or tuning kernels comes after physical correctness and performance
measurements. Publish reports with the exact LeatEngine commit and host configuration.
