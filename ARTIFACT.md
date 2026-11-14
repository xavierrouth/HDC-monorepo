# ApproxHDC — Build, Run, and Reproduce

## Prerequisites

- Linux (tested on Ubuntu 20.04 / 22.04)
- Python 3.11
- NVIDIA GPU with OpenCL support (for GPU targets)
- ~10 GB disk for HPVM build + datasets

## 1. Build HPVM-HDC Compiler

ApproxHDC requires the `hpvm-hdc-25` branch of HPVM-HDC.

HPVM-HDC must be built in both debug and release modes. The release
build is used for all tuning runs.

```bash
# After building HPVM, verify:
ls hpvm/build-release/bin/clang++        # release compiler
ls hpvm/build-release/lib/LLVMHPVMHDC.so # HDC passes
```

## 2. Clone and Set Up HDC-Monorepo

```bash
git clone https://github.com/xavierrouth/HDC-monorepo.git
cd HDC-monorepo
```

### Python Environment

The tuner requires a Python 3.11 virtual environment with OpenTuner
and dependencies:

```bash
python3.11 -m venv approx-tuner/venv
source approx-tuner/venv/bin/activate
pip install opentuner numpy pandas matplotlib
```

**Always activate the venv before running any scripts:**

```bash
source approx-tuner/venv/bin/activate
```

### Datasets

Datasets are stored in `datasets/` and within individual application
directories:

| Application | Dataset | Location |
|---|---|---|
| HD-Classification | ISOLET (spoken letters, 26 classes) | `datasets/classification/isolet_nq_{train,test}{X,Y}.bin` |
| HD-Clustering | ISOLET | `datasets/clustering/isolet_train_{trainX,trainY}.bin` |
| HD-Graph (RelHD) | Cora (citation network, 7 classes) | Built into `applications/HD-Graph/` |
| HD-HashTable | small_bacillus.fastq (genomic k-mers) | `applications/HD-HashTable/dataset/` |
| HD-Sim-Classification | ISOLET (via SpecPCM simulator) | `datasets/classification/` |
| Micro-HD-Classification | ISOLET | `datasets/classification/` |

### Verify a Baseline Build

Confirm the HPVM compiler can build and run an application:

```bash
cd applications/HD-Classification
make host-gpu-approx HPVM_BUILD=release ENCODING_DIM=2048
./host-gpu-approx
# Should print accuracy ~0.87-0.90 and write results to results.txt
cd ../..
```

## 3. Running Tuning Experiments

All tuning experiments are launched through `results/scripts/eval_redperf.py`.
This script wraps per-application `autotune.py` files and manages
the tuning series, database naming, and plot generation.

### Quick Reference

```bash
source approx-tuner/venv/bin/activate

# Single app, single device, 3-hour budget per series:
python3 results/scripts/eval_redperf.py \
    --app classification --device gpu --stop-after 10800

# All apps, both devices (long — runs every series sequentially):
python3 results/scripts/eval_redperf.py \
    --app all --device gpu --stop-after 10800

# Run specific series only:
python3 results/scripts/eval_redperf.py \
    --app classification --device gpu \
    --series-filter redperf_endonly,redperf_off \
    --stop-after 10800
```

### CLI Arguments

| Flag | Description | Default |
|---|---|---|
| `--app` | Application: `clustering`, `classification`, `graph`, `hashtable`, `all` | required |
| `--device` | Hardware target: `cpu` or `gpu` | `cpu` |
| `--stop-after` | Tuning budget per series in seconds | 18000 (5h) |
| `--accuracy-target` | QoS target passed to OpenTuner | per-app default |
| `--accuracy-cutoff` | Override plot cutoff (filters configs below this) | same as target |
| `--series-filter` | Comma-separated series suffixes to run | all series |
| `--plot-only` | Skip tuning; regenerate plots from existing DBs | off |
| `--plot-matrix` | Generate the multi-app matrix plot (both orientations) | off |
| `--annotate-configs` | Annotate speedup plot with config numbers + dump decoded configs | off |
| `--baseline-override` | Use per-app baseline overrides instead of seed_time | off |

### Tuning Series

Each application runs up to four tuning conditions:

| Series | Description | Search Space Impact |
|---|---|---|
| `redperf_on` (Full RedPerf) | All three RP knobs (start, stride, end) on every HDC op | Largest |
| `redperf_endonly` (End-only + hint) | Only end/extent knob; start=0, stride=1 fixed. Encoder ops protected by no-approx hint. | Medium |
| `redperf_off` (No RedPerf) | No RP knobs; only D, algorithmic, and custom knobs | Smallest |
| `redperf_on_hint` (Full RedPerf + hint) | All RP knobs + encoder no-approx hint | Large |

HD-HashTable and HD-Graph (RelHD) run only three series (no +hint,
as those apps have no approximatable encoding intrinsics).

### Output

Tuning results are written to SQLite databases in `results/dbs/`:

```
results/dbs/hd_classification_redperf_on_20260414_155333.db
results/dbs/hd_classification_redperf_on_20260414_155333.db.device
results/dbs/hd_classification_redperf_on_20260414_155333.db.searchspace
```

Each `.db` file is a standard OpenTuner database. The `.device` and
`.searchspace` sidecars record the target device and search space size.

## 4. Generating Plots

### Matrix Plot (Main Figure)

Generates the multi-app speedup-over-time matrix shown in the paper
(both horizontal and vertical orientations):

```bash
python3 results/scripts/eval_redperf.py --plot-matrix
```

Output:
- `results/attachments/redperf_matrix.pdf`
- `results/attachments/redperf_matrix_vertical.pdf`
- `results/attachments/redperf_search_spaces.txt`
- `results/attachments/redperf_knobs.txt`

### Per-App Plots

```bash
# Single app speedup + CDF plots:
python3 results/scripts/eval_redperf.py --app clustering --plot-only

# With config annotations and decoded knob dump:
python3 results/scripts/eval_redperf.py --app clustering --plot-only --annotate-configs
```

### MicroHD Comparison Plot

```bash
python3 results/scripts/compare_microhd.py --plot-only
```

Uses existing data from `results/dbs/micro_hd_classification_opentuner.db`
and `results/attachments/microhd_sweep_*_hpvm.csv`. Output:
`results/attachments/microhd_vs_approxhdc_relative_*.pdf`

### SpecPCM Plot

```bash
python3 results/scripts/eval_sim.py --plot-only
```

Output: `results/attachments/hd_sim_classification_*_speedup.pdf`

## 5. Reproducing Paper Results

### Table: Search Space Sizes

Generated automatically by `--plot-matrix`:
`results/attachments/redperf_search_spaces.txt`

### Figure: Speedup Matrix (Figure 2)

```bash
python3 results/scripts/eval_redperf.py --plot-matrix
# -> results/attachments/redperf_matrix.pdf
```

To reproduce from scratch (runs all 4 apps × 2 devices × 3-4 series
× 3h each = ~72-96 hours total):

```bash
for device in gpu cpu; do
    python3 results/scripts/eval_redperf.py \
        --app all --device $device --stop-after 10800
done
python3 results/scripts/eval_redperf.py --plot-matrix
```

### Figure: MicroHD Comparison (Figure 3)

To replot from existing data:
```bash
python3 results/scripts/compare_microhd.py --plot-only --accuracy-target 0.845
```

To reproduce from scratch (requires MicroHD + torchhd installed):
```bash
python3 results/scripts/compare_microhd.py \
    --device gpu --budget 7200 --accuracy-target 0.845 \
    --microhd-python /path/to/torchhd-venv/bin/python3
```

### Figure: SpecPCM Tuning (Figure 4)

To replot from existing data:
```bash
python3 results/scripts/eval_sim.py --plot-only
```

To reproduce from scratch:
```bash
python3 results/scripts/eval_sim.py --stop-after 10800
```

## 6. Project Structure

```
HDC-monorepo/
├── applications/
│   ├── HD-Classification/     # autotune.py, Makefile, src/, config/
│   ├── HD-Clustering/
│   ├── HD-Graph/              # (labeled "RelHD" in plots)
│   ├── HD-HashTable/
│   ├── HD-Sim-Classification/ # SpecPCM simulator target
│   └── Micro-HD-Classification/
├── approx-tuner/
│   ├── venv/                  # Python 3.11 virtualenv
│   └── hdc_approximation_tuner/  # Harness, ApproxTypes, PCMDesc
├── datasets/
│   ├── classification/        # ISOLET binary files
│   └── clustering/            # ISOLET, MNIST, UCIHAR binary files
├── results/
│   ├── scripts/               # eval_redperf.py, eval_sim.py, etc.
│   ├── dbs/                   # OpenTuner SQLite result databases
│   └── attachments/           # Generated plots (PDF)
```

tuning experiment first.
