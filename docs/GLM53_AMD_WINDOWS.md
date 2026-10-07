# GLM-5.3-Flash on AMD / Windows

Notes from running this branch (PR #1315, glm5-next) on an RX 7900 XTX under Windows. No source change was needed.

Machine: RX 7900 XTX (gfx1100, 24 GB, driver 32.0.31041.1004), Ryzen 9 7900X3D, 64 GB RAM, Windows 11 24H2.
ROCm comes from the TheRock wheels pinned by the build script (10.2.0a20260930).

## Build

```bat
set STRATA_HIP_ARCHS=gfx1100
tools\hip\build_windows.bat tests
```

Builds clean. The only warnings are unused `hipError_t` returns, none in the glm files.

## glm_parity

```bat
set HIP_VISIBLE_DEVICES=1
build-hip-win\glm_parity.exe --selftest
```

115/115 on the 7900 XTX.

The 7900X3D has an integrated Radeon and HIP lists it as device 0, the 7900 XTX as device 1. If you forget
`HIP_VISIBLE_DEVICES`, the test runs on the iGPU, every glm kernel fails with `device kernel image is invalid` and it
reports 50 failures out of 115. That is not a math problem, the build simply has no code for the iGPU. The engine is
not affected because setup passes `"gpu": 1`.

## End to end

Unsloth UD-IQ2_XXS (4 shards, 101.8 GB) on 24 GB VRAM + 64 GB RAM. The expert set is 92 GiB, so it does not fit
in RAM: the engine runs with `--mmap-experts`, and `--prefill 1` because the chunked prompt read needs sliceable
experts and this file's IQ2_XXS gate/up are not.

Two things this branch adds on top of the PR:

- `--glm-gpu-experts 0`: a per-layer set of experts in VRAM, computed on the card, the CPU pool only gets the
  misses. Slots fill as experts are routed, then LFU decides who stays.
- thinking off now works for GLM. Its template always opens `<think>` and maps unknown efforts to `max`, so
  `reasoning_effort: "none"` used to think at full length.

Decode, 256 tokens, same prompt five times in a row:

| | tok/s |
|---|---|
| llama.cpp Vulkan, 5 layers of experts on the card (`-ncmoe 40`) | 4.2 to 4.7 |
| this PR, experts on the CPU | 2.5 |
| `--glm-gpu-experts 0` (13.8 GiB, 46 slots a layer) | 2.9, 4.1, 5.9, 4.8, 5.3 |

About 71% of the routed experts are hits once warm. `STRATA_GLM_GPU_CHECK=1` recomputes every hit on the CPU:
the gap is about 2% of the largest value, from the activation rounding (q8_1 on the card, q8_K on the CPU).

Prompt reading is the weak spot: about 2.4 tok/s token by token, against 3.8 for llama.cpp. A chunked read
with the experts streamed to the card is the next thing to do.
