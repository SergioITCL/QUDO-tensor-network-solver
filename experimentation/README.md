# Experiment reproducibility

This directory contains the scripts that generate experimental results and the processors that convert JSON results into CSV, Markdown, LaTeX tables, and figures.

## Shared configuration

All experiment parameters are stored in [`experiments.json`](experiments.json). Edit that file to change problem sizes, `(d, k)` configurations, numbers of instances, solver limits, or result directories. Individual experiment scripts load their section from this file.

## Run all experiments

The following command runs every experiment in JSON order and executes its processor immediately afterwards:

```bash
poetry run python experimentation/run_all_experiments.py
```

The full campaign can take a long time. Experiments 2, 6, and 7 are the most expensive. To run a single experiment, use the commands in its section below.

Run all commands from the repository root:

```bash
cd /path/to/QUDO-tensor-network-solver
poetry install
```

Optional environment checks:

```bash
poetry run python -c "import numpy, scipy, torch, tensorkrowch; print('core dependencies available')"
poetry run python -c "from pyscipopt import Model; print('SCIP available')"
```

For every reproduction, record `git rev-parse HEAD`, `python --version`, dependency versions, CPU, RAM, operating system, and start/end timestamps. Runtime and memory measurements depend on the hardware and numerical backend.

## Instances and seeds

Instances are generated with `qudo_problem_generation` and the explicit integer lists in the `seeds` field of `experiments.json`. Quadratic coefficients are sampled from `U(-10, 10)` and linear coefficients use an independent sequence. `q_matrix` uses a compact lower-triangular representation: each row ends with the diagonal coefficient and contains interactions with previous variables.

The workflow for every experiment is:

```text
experiment script → results/*.json → processor → processed_results/*
```

Generated JSON files are excluded from Git. Processors therefore require the corresponding experiment script to have been run first, unless raw results are obtained from another artifact.

## Experiment 1: SMVC accuracy

Files: `experiment_1_accuracy/experiment_1_accuracy.py` and `experiment_1_accuracy/experiment_1_processor.py`.

Configuration: `n ∈ {500,1000}`, `(d,k) ∈ {(2,2),(2,4),(4,2),(4,4)}`, and 50 random instances per configuration.

```bash
poetry run python experimentation/experiment_1_accuracy/experiment_1_accuracy.py
poetry run python experimentation/experiment_1_accuracy/experiment_1_processor.py
```

Outputs include the raw JSON and `experiment_1_accuracy.csv`/`.tex`.

## Experiment 2: method comparison

Files: `experiment_2_heuristics_comparasion/experiment_2_heuristics_comparasion.py` and `experiment_2_processor.py`.

Configuration: `n ∈ {250,500}`, `d ∈ {2,4,6}`, `k ∈ {2,3,4}`, and 50 instances per configuration. Beam DP calibrates its width against SMVC runtime; Tabu Search and SCIP receive an approximately matched budget.

```bash
poetry run python experimentation/experiment_2_heuristics_comparasion/experiment_2_heuristics_comparasion.py
poetry run python experimentation/experiment_2_heuristics_comparasion/experiment_2_processor.py
```

Outputs include one JSON per configuration, `summary_by_d_k_n.csv`/`.md`, `experiment_2_comparison.tex`, and two PNG figures.

## Experiments 3, 4, and 5: scaling

These experiments compare SMVC, STC, and Exact Dynamic Programming. Each point uses three random instances.

### Experiment 3

Fixes `n=100`; varies `k=1..14` with `d=2`, and `d=2..29` with `k=2`.

```bash
poetry run python experimentation/experiment_3_kd_t/experiment_3_kd_t.py
poetry run python experimentation/experiment_3_kd_t/experiment_3_processor.py
```

### Experiment 4

Fixes `k=2`; varies `n=200,400,600,800,1000` and `d=2..8`.

```bash
poetry run python experimentation/experiment_4_n_vs_t_d/experiment_4.py
poetry run python experimentation/experiment_4_n_vs_t_d/experiment_4_processor.py
```

### Experiment 5

Fixes `d=2`; varies `n=200,400,600,800,1000` and `k=2..10`.

```bash
poetry run python experimentation/experiment_5_n_vs_t_k/experiment_5.py
poetry run python experimentation/experiment_5_n_vs_t_k/experiment_5_processor.py
```

Each experiment writes a JSON file under `results/` and a PNG figure plus LaTeX table under `processed_results/`.

## Experiment 6: SCIP until reaching SMVC

Files: `experiment_6_scip_time_to_tn/experiment_6_scip_time_to_tn.py` and `experiment_6_processor.py`.

Configuration: `n=500`, `(d,k) ∈ {(2,2),(2,4),(4,4),(6,4)}`, 50 instances per configuration, and a 60-second SCIP limit per instance. The script writes per-configuration JSON checkpoints.

```bash
poetry run python experimentation/experiment_6_scip_time_to_tn/experiment_6_scip_time_to_tn.py
poetry run python experimentation/experiment_6_scip_time_to_tn/experiment_6_processor.py
```

Outputs include `configuration_summary.csv`, `instance_results.csv`, Markdown, and LaTeX tables.

## Experiment 7: memory

Files: `experiment_7_memory/experiment_7_memory.py` and `experiment_7_processor.py`.

RSS is measured in child processes using three seeds per configuration:

- vary `n=100..1000`, with `k=3,d=3`;
- vary `k={2,4,6,8,10}`, with `n=50,d=3`;
- vary `d={2,4,6,8,10}`, with `n=50,k=3`;
- Beam DP width 256 and Tabu Search with 100 iterations;
- 2 ms RSS sampling interval.

```bash
poetry run python experimentation/experiment_7_memory/experiment_7_memory.py
poetry run python experimentation/experiment_7_memory/experiment_7_processor.py
```

Outputs include `experiment_7_memory.csv`, `.tex`, and `.png`. This experiment can require substantial time and memory.

## Verification checklist

Before accepting a reproduction, check that every JSON contains the expected number of instances, seeds and parameters match the configuration, costs use the same quadratic convention, and tables are generated automatically from JSON rather than edited manually. Runtime and RSS differences should only be compared under equivalent hardware conditions.

## Current limitations

- Raw results are not versioned in Git.
- No container or CI workflow validates the full campaign.
- Experiments 6 and 7 can be expensive.
