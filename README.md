# HDC-monorepo

ApproxHDC: automated approximation tuning for Hyperdimensional Computing
workloads. The tuner searches over approximation knobs — encoding
dimensionality, element type, similarity metric, and reduction perforation —
and reports configurations that meet a per-application quality-of-service
target on CPU, GPU, and simulated ReRAM/PCM targets.

Build, run, and reproduction instructions are in [ARTIFACT.md](ARTIFACT.md).

## Layout

- `applications/` — the HDC benchmarks (classification, clustering, graph,
  hash table) written in HDC++
- `approx-tuner/` — the OpenTuner-based search harness
- `results/scripts/` — evaluation and plotting scripts
- `datasets/` — preprocessed input data

## HPVM version

ApproxHDC builds against the `hpvm-hdc-25` branch of HPVM-HDC.
