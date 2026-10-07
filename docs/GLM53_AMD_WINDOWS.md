# GLM-5.3-Flash (glm5-next) on AMD / Windows — validation notes

Branch `glm53-hip` = PR #1315 (glm5-next: GLM-5.3-Flash support), built and tested on an AMD card on Windows.
No source change was needed for the HIP build.

## Machine

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (gfx1100, 24 GB), driver 32.0.31041.1004 |
| CPU | AMD Ryzen 9 7900X3D (has an integrated Radeon GPU, see below), 64 GB RAM |
| OS | Windows 11 Pro 24H2 |
| ROCm | TheRock wheels 10.2.0a20260930 (the version `tools\hip\build_windows.bat` pins) |

## Build

```bat
set STRATA_HIP_ARCHS=gfx1100
tools\hip\build_windows.bat tests
```

Builds clean (`strata.exe`, tests, `dist\strata-windows-x64-hip.zip`); only `-Wunused-value` warnings on ignored
`hipError_t` returns, none in the glm files.

## glm_parity

```bat
set HIP_VISIBLE_DEVICES=1
build-hip-win\glm_parity.exe --selftest
```

**115 / 115 cases pass on the RX 7900 XTX** (mHC, KDA gate / conv / l2norm / delta, MLA attention, every rival
told apart).

**Gotcha — integrated GPU first:** on this machine HIP lists the Ryzen's integrated Radeon as device 0 and the
7900 XTX as device 1. Run without `HIP_VISIBLE_DEVICES`, `glm_parity` picks device 0, every glm kernel launch fails
with `device kernel image is invalid` (the build has no code for the iGPU) and the self-test reports
`115 cases, 50 failures`. These are not arithmetic failures. The engine itself is unaffected (setup's `"gpu": 1`
selects the right card); only the stand-alone tests use device 0.

## End-to-end run

Pending: Unsloth `UD-IQ2_XXS` (4 shards, 101.8 GB) on 24 GB VRAM + 64 GB RAM, compared with llama.cpp (Vulkan) on the
same machine and the same prompts. Results will be added here.
