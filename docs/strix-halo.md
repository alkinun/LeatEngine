# Strix Halo bring-up

The initial target is a Ryzen AI Max+ 395 system with 128 GB unified memory and a Radeon
8060S (`gfx1151`, RDNA 3.5). The machine has been ordered but has not yet been tested.
This document is a qualification procedure, not a claim of measured support or performance.

## Current evidence

| Execution path | Evidence |
|---|---|
| CPU reference | Generated-model correctness tests |
| AMD | Kernel and tiny-model tests on tinygrad's RDNA 3 emulator |
| NVIDIA | Inherited RTX 3090 measurements in the benchmark document |
| Physical Strix Halo | Pending |

The current kernels select their RDNA 3 path for architecture major 11. Whether this gives
correct, efficient execution on `gfx1151` must be confirmed on the physical machine.

## Prepare the host

Use Linux with firmware and an amdgpu driver that support the device. Install uv, Git, and
the ROCm comgr compiler required by tinygrad's current AMD path. Ensure the user can access
`/dev/kfd` and the relevant render device; device permissions depend on the distribution.

Record the distribution, kernel, firmware/BIOS, compiler, memory configuration, and power mode.
Inspect available GPU memory before selecting models or contexts. Firmware UMA allocation and
Linux TTM/GTT limits can affect usable memory; verify their behavior on this machine before
applying tuning. No kernel-memory settings are assumed validated here.

```bash
uv sync --locked
DEV=AMD uv run python -c "from tinygrad import Device; print(Device['AMD'].arch)"
DEV=AMD uv run pytest -m gpu
DEV=AMD uv run python scripts/validate.py /models --llama-cpp /path/to/llama.cpp/build/bin
```

Run the real-GPU tests in a single process. Start with a model that comfortably fits memory,
then increase model size, context, and simultaneous sequences. Test vision, embeddings, and
speculation with their matching weights before describing them as hardware-verified.

## Qualification

1. Confirm device detection, allocation, kernel correctness, and stable repeated generation.
2. Compare model quality against reference execution and llama.cpp using matching GGUF files.
3. Measure prefill/decode, first-token latency, and memory at documented contexts and slot counts.
4. Exercise concurrent requests, cancellation, model switching, vision, and embeddings.
5. Repeat longer runs to expose memory pressure, thermal effects, and startup/compile costs.

For llama.cpp comparisons, record the exact commit and build options for each backend, such as
Vulkan or HIP. Keep model files, quantization, prompt lengths, and generation settings comparable;
report differences rather than mixing them into one headline number.

The emulator's clang compiler adapter is test tooling. Promoting a clang path to the runtime,
changing memory defaults, or tuning kernels comes after physical correctness and performance
measurements. Publish reports with the exact LeatEngine commit and host configuration.
