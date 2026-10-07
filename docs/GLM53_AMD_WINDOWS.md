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

In progress: Unsloth UD-IQ2_XXS (4 shards, 101.8 GB) on 24 GB VRAM + 64 GB RAM, against llama.cpp Vulkan on the same
machine with the same prompts. Numbers will go here.
