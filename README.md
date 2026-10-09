# A Two-Stage Genetic Algorithm for 3D PCB Placement and Routing

**Optimization of Layout Score and Computational Complexity**

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-parallel%20evaluation-2C2255)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Final Year Project](https://img.shields.io/badge/Final%20Year%20Project-XMUM-orange)

> **Final Year Project** — Department of Physics, Xiamen University Malaysia (XMUM)
> **Author:** Tan Kok Jing (PHY2304248) · **Supervisor:** Prof. Kelvin Ooi Jian Aun · June 2026

---

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Empirical Highlights](#empirical-highlights)
- [Repository Structure](#repository-structure)
- [Environment & Prerequisites](#environment--prerequisites)
- [Installation & Build Instructions](#installation--build-instructions)
- [Usage Guide](#usage-guide)
- [Inputs & Outputs Specification](#inputs--outputs-specification)
- [Citation](#citation)
- [License](#license)

---

## Overview

Placement and routing (P&R) on a printed circuit board is an **NP-hard combinatorial optimization problem**: the feasible search space grows factorially with the number of movable components, while additional nets and routing layers expand the solution space even faster. Conventional single-layer genetic algorithms — which encode every component as an independent gene — suffer from **premature convergence**, local optima, and an `O(N!)`-scale chromosome search that wastes most of its evaluations on layouts where strongly connected components have been scattered apart.

This project presents a **two-stage genetic algorithm** implemented in a high-performance **C++ engine** (exposed to Python through `pybind11`) for the **joint optimization of 3D PCB placement and routing**. The key idea is that *search-space reduction should follow the topology of the netlist, not an arbitrary decomposition*: strongly coupled components are first extracted into rigid local structures, the macroscopic arrangement of those structures is optimized with cheap geometric surrogates, and only then is the full routing-aware objective applied to refine individual component positions with a 3D A* router, congestion history, and partial rip-up & reroute.

<iframe src="./images/fig1_two_stage_ga_architecture.pdf" width="100%" height="600px"></iframe>

The optimization *fitness* (with generation-dependent heuristic weights) is deliberately decoupled from the **standardized Layout Score** (with constant weights, re-routed under a fixed configuration), so that different algorithmic variants can be compared in a single coordinate system.

---

## Key Innovations

### 1. Netlist-Aware Topology Decomposition (Super-Nodes)

Groups are extracted automatically from the netlist graph. Every component is a vertex, and every net `N_k` joining `n_k` components is treated as a complete clique whose pairs receive the inverse-cardinality weight

$$
W_{ij} \;=\; \sum_{k \,\in\, \mathrm{Nets}(i,j)} \frac{1}{|n_k| - 1}.
$$

A two-pin net contributes the maximum weight `1` (strongest local binding), whereas a high-cardinality power/ground net contributes only `1/(|n_k|-1)`. Edges are processed in descending order and merged greedily while respecting a maximum group size, producing **rigid super-nodes**. The Stage I chromosome then encodes only a transform and a candidate index per group, `τ_g = (x_g, y_g, θ_g, k_g)`, reducing the chromosome from `n` component genes to `q ≪ n` group genes and eliminating entire volumes of invalid search space. Group extraction is dominated by the edge sort, `O(|E| log |E|)`.

### 2. Ghost-Pin Guide Strategy

Nets that leave a group attach a fixed virtual target — a **ghost pin** `g_s ∈ ∂Ω_G` — on the boundary of the group's placement square, distributed round-robin over the four edges. The centripetal term

$$
D_{cen} \;=\; \sum_{s \in S} \lVert p_s - g_s \rVert_2
$$

pulls externally connected pins toward a definite exit direction instead of letting them drift to the group center. When the locally optimized constellations are later combined at the global level, **cross-group wires are already aligned**, which substantially shortens the Stage II closure and narrows the score band of the convergence curve.

### 3. Decoupled Two-Stage Optimization

| Aspect | Stage I — Group level | Stage II — Component level |
| --- | --- | --- |
| Genes | `τ_g = (x_g, y_g, θ_g, k_g)` | `g_i = (x_i, y_i, θ_i)` |
| Evaluation | Cheap placement surrogates `F_I` (MST pin distance, overlap, repulsion, ghost-pin centripetal distance), switching to routing-aware evaluation after `out_routing_start_gen` | Full routing-aware fitness `F_II` (wire length, parasitic coupling, crossings, wire–component overlap, vias) |
| Search | Group transforms + **softmax candidate selection** over EMA scores with exponentially annealed temperature `T(t) = T₀(T_end/T₀)^{t/G}` | Gaussian component-level mutation with linearly decaying step `σ(t)`, uniform crossover |
| Routing | Group-local A*; cross-group nets represented by ghost pins | **3D A* on a layered occupancy grid** + congestion-history rip-up & reroute + **incremental partial rip-up & reroute** (`O(N_affected)` nets per mutation instead of `O(N_nets)`) |

This division of labour — cheap macro discovery followed by expensive micro refinement — prevents premature convergence and removes the factorial chromosome growth of conventional single-layer GAs.

### 4. Computational Efficiency & Layout Score

The two-stage engine parallelizes evaluation across individuals with **OpenMP**, routes on a **3D A\* grid** with layer-direction preferences and a capped inflation heuristic, and periodically performs a **full re-route** to purge residual incremental artifacts. A final standardized scoring pass re-routes the best individuals under a fixed configuration (`ρ = 1.0 mm`, `V = 3.5`, wire–component penalty `400`, crossing penalty `1500`, overlap penalty `6000`), enabling fair, hyperparameter-independent benchmarking.

---

## Empirical Highlights

Evaluated on six synthetic benchmark complexities (13.0 → 28.8, twenty runs each; single-layer GA baseline as reference), both objectives — **layout quality** and **computational cost** — improved simultaneously rather than being traded off:

| Benchmark complexity | Two-stage Layout Score | Single-layer Layout Score | Relative difference | Time ratio |
| --- | --- | --- | --- | --- |
| 13.0 | 549.97 | **280.70** | +95.9% | 0.53 |
| 16.2 | 1086.36 | **794.08** | +36.8% | 0.58 |
| 19.1 | **1558.29** | 1695.75 | −8.1% | 0.45 |
| 22.0 | **1797.99** | 3245.39 | −44.6% | 0.53 |
| 25.4 | **2722.87** | 5524.27 | **−50.7%** | 0.46 |
| 28.8 | **5174.51** | 8970.86 | −42.3% | **0.38** |

- **Up to 50.7% lower mean Layout Score** beyond the crossover region (19.1–28.8).
- **Only 38%–58% of the baseline's wall-clock time** across the whole range; the gap widens with complexity (mean runtimes at 28.8: **235.0 s vs. 611.7 s**).
- **Sub-exponential runtime growth** with benchmark complexity, while the baseline diverges increasingly.
- **More consistent runs**: at complexity 28.8 the baseline's score standard deviation is `4770.09` versus `1875.93` for the two-stage GA.

---

## Repository Structure

```text
.
├── src/
│   ├── pcb_engine.cpp          # C++ core: two-stage GA, 3D A* router, ghost-pin
│   │                           # heuristics, rip-up & reroute, OpenMP acceleration
│   ├── main_wrapper.py         # Python entry point: loads JSON configs, calls
│   │                           # pcb_engine.run_ga_optimization(), saves JSON/CSV/PNG/logs
│   └── param_generator_ui.py   # Tkinter UI to configure all GA parameters and export JSON
├── tools/
│   └── PCB_connect_generator.py# Synthetic benchmark generator: builds scalable
│                               # netlists with a parametric complexity measure
├── benchmarks/
│   ├── sample_connect.json     # Sample connectivity/topology input (components, nets, groups)
│   └── sample_config.json.json # Sample global parameter configuration input
├── output_sample/
│   ├── json/                   # best_XX.json (top-N layouts), group_best_XX.json
│   ├── csv/                    # fitness_history, fitness_components, layout_scores,
│   │                           # group_fitness/ (per-group GA convergence data)
│   ├── png/                    # convergence plots, rendered layouts, component plots
│   └── logs/                   # run_summary.json, terminal_log.txt
├── setup.py                    # Build script: compiles pcb_engine.cpp into the
│                               # Python C++ extension module (pcb_engine.pyd / .so)
├── requirements.txt            # Python dependencies
└── README.md
```

> The compiled extension (`pcb_engine.pyd`, `pcb_engine.*.so`, or `build/`) is generated
> locally and is not part of the repository.

---

## Environment & Prerequisites

| Requirement | Details |
| --- | --- |
| **C++ compiler** | GCC/Clang or MSVC with **C++17** support |
| **OpenMP** | Required for multi-threaded evaluation (`-fopenmp` on GCC/Clang, `/openmp` on MSVC) |
| **Python** | **3.8+** (3.10+ recommended) with `pip` |
| **Python packages** | See [`requirements.txt`](requirements.txt) — `setuptools`, `pybind11`, `numpy`, `matplotlib`, `shapely` |
| **Tkinter** | Bundled with CPython on Windows/macOS; on Debian/Ubuntu run `sudo apt-get install python3-tk` (needed only for `param_generator_ui.py`) |

**Platform notes**

- **Windows:** Visual Studio 2019/2022 with the *Desktop development with C++* workload (MSVC provides OpenMP via `/openmp`).
- **Linux (GCC):** `sudo apt-get install build-essential python3-dev`. GCC ships OpenMP support out of the box.
- **Linux/macOS (Clang):** install `libomp` (e.g., `brew install libomp` on macOS) since Clang requires it for `-fopenmp`.

---

## Installation & Build Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build the C++ extension module

`setup.py` compiles `pcb_engine.cpp` and, because the extension sources are resolved
relative to the **current working directory**, the build must be started inside `src/`
(where `pcb_engine.cpp` lives). With `--inplace`, the compiled module lands directly in
`src/` so that `main_wrapper.py` can import it.

```bash
cd src
python ../setup.py build_ext --inplace
cd ..
```

- **Windows (MSVC):** produces `src/pcb_engine.cp<ver>-win_amd64.pyd`
- **Linux/macOS (GCC/Clang):** produces `src/pcb_engine.cpython-<ver>-<arch>.so`

Both are imported simply as `import pcb_engine`.

### 4. Verify the build

```bash
cd src
python -c "import pcb_engine; print(pcb_engine.run_ga_optimization.__doc__)"
cd ..
```

Expected output:

```text
run_ga_optimization(arg0: dict) -> dict

Run Two-Stage GA for PCB placement and routing
```

> **Optional:** limit the OpenMP worker threads with the standard environment variable,
> e.g. `set OMP_NUM_THREADS=8` (Windows) or `export OMP_NUM_THREADS=8` (Linux/macOS).

---

## Usage Guide

The complete workflow is: **generate a benchmark → create a parameter file → run the optimizer**.

### Step 1 — Generate synthetic test topologies (optional)

The generator builds benchmark circuits with a controlled topological complexity
`Complexity = α|V| + β|E| + γ|P|/|E|` (`α = 1.2`, `β = 0.5`, `γ = 2.0`), repairs
degenerate nets, and groups strongly coupled components automatically.

Edit the `CONFIG` dictionary at the top of `tools/PCB_connect_generator.py`
(component limits, complexity targets, `output_directory`, `num_jsons_to_generate`), then run:

```bash
python tools/PCB_connect_generator.py
```

Each generated `layout_XX.json` is a valid connectivity/topology input for the next steps.

### Step 2 — Generate or modify the parameter configuration (optional)

Launch the Tkinter parameter editor:

```bash
python src/param_generator_ui.py
```

The UI exposes every GA parameter (grouping, Stage I/II schedules, penalties, diversity
control) with tooltips documenting its effect, and exports a flat JSON file
(e.g., `custom_params.json`). Any subset of keys may be provided — unspecified keys
fall back to the defaults in `src/main_wrapper.py`. A ready-made example is
included in `benchmarks/sample_config.json.json`.

### Step 3 — Run the placement & routing optimization

```bash
python src/main_wrapper.py \
    -c benchmarks/sample_connect.json \
    -o output \
    -p benchmarks/sample_config.json.json
```

| Argument | Required | Description |
| --- | --- | --- |
| `-c`, `--config` | Yes | Input connectivity/topology JSON (components, pins, nets, groups) |
| `-o`, `--output` | Yes | Output root directory (created automatically) |
| `-p`, `--params` | No | External global-parameter JSON; overrides built-in defaults |

The engine prints progress to the console while mirroring everything to
`<output>/logs/terminal_log.txt`. A full run on the sample benchmark completes in a few
minutes; for a quick smoke test, reduce `group_pop_size` (~30–50) and
`out_generations` (~100–300) in the parameter JSON.

---

## Inputs & Outputs Specification

### Input 1 — Global parameter JSON (flat key–value)

All keys are optional and are merged over the defaults. Example (excerpt):

```json
{
  "num_layers": 2,
  "grid_res": 1.0,
  "wirelength_weight": 1.0,
  "capacitance_weight": 1.5,
  "via_penalty": 3.5,
  "group_pop_size": 300,
  "group_generations": 500,
  "out_pop_size": 300,
  "out_generations": 3000,
  "out_combine_start_gen": 800,
  "layout_score_cap_weight": 1.5,
  "layout_score_wire_weight": 1.0,
  "layout_score_congestion_weight": 5.0
}
```

Key groups: **shared weights & penalties**, **group GA** (`group_*`), **two-stage global GA**
(`out_*`, `softmax_*`, `stage2_*`), and **diversity control** (`diversity_*`). See
`src/param_generator_ui.py` for the full annotated list.

### Input 2 — Connectivity / topology JSON

```json
{
  "metadata": {
    "V_score": 14.4, "E_score": 4.0, "P_score": 7.25,
    "V_count": 12, "E_count": 8, "P_count": 29
  },
  "pcb_bounds": { "xmin": 0.0, "ymin": 0.0, "xmax": 300.0, "ymax": 300.0 },
  "components": [
    {
      "id": "C1",
      "width": 8.95,
      "height": 10.3965,
      "pins": [ { "name": "1", "dx": -2.5375, "dy": 0.0 },
                { "name": "2", "dx":  2.5375, "dy": 0.0 } ]
    }
  ],
  "groups": [
    { "group_name": "AutoGroup_1", "members": ["C1", "R1", "U1"] }
  ],
  "nets": [
    { "net_id": "Net-Construct_1",
      "connections": [ { "comp_id": "C1", "pin_name": "1" },
                       { "comp_id": "R1", "pin_name": "2" } ] }
  ]
}
```

- `components[].pins[].dx/dy` are the pin offsets relative to the component center (mm); rotation is restricted to `{0°, 90°, 180°, 270°}`.
- `pcb_bounds` defines the placeable region; `groups` may be omitted and will be derived automatically.
- `metadata` records the benchmark complexity `V/E/P` scores produced by the generator.

### Outputs

Running `main_wrapper.py` creates the following structure under `--output`:

| Path | Contents |
| --- | --- |
| `json/best_XX.json` | Top-N optimized individuals: metadata (`fitness`, `total_wire_length`, `total_cap`, `overlap_score`, `wire_cross_count`), footprints, nets, tracks and vias (KiCad 10.0-oriented export) |
| `json/group_best_XX.json` | Best local constellation of each group from the group-level GA |
| `csv/fitness_history.csv` | Best `fitness` per generation |
| `csv/fitness_components.csv` | Per-generation breakdown: wire length, capacitance, overlap, crossings, wire–component overlap, pin distance, repulsion penalty |
| `csv/layout_scores.csv` | Best **Layout Score** per generation (from the standardized scoring pass) |
| `csv/group_fitness/group_fitness_XX.csv` | Per-group GA convergence history |
| `png/fitness_history.png`, `png/layout_scores.png` | Global convergence curves (`fitness` and `layout_score`) |
| `png/best_XX.png`, `png/group_best_XX.png` | Rendered layout & routing diagrams (component outlines, pads, layer-styled tracks, vias) |
| `png/fitness_components/*.png`, `png/group_fitness/*.png` | Individual term convergence plots |
| `logs/run_summary.json` | Run metadata, elapsed time, and the effective `GLOBAL_PARAMS` |
| `logs/terminal_log.txt` | Complete console transcript of the run |

The standardized metric reported in every result is

$$
\mathrm{LayoutScore} \;=\; 1.5\,C_{total} \;+\; 1.0\,L_{wire} \;+\; 5.0\,P_{congestion},
\qquad P_{congestion} = 10\,N_{cross} + S_{overlap},
$$

evaluated in an independent final pass with a fixed routing configuration.

---

## Citation

If you use this code or build upon this work, please cite the project report:

```bibtex
@thesis{tan2026twostage,
  author       = {Tan, Kok Jing},
  title        = {A Two-Stage Genetic Algorithm for {3D} {PCB} Placement and
                  Routing: Optimization of Layout Score and Computational Complexity},
  school       = {Xiamen University Malaysia},
  year         = {2027},
  type         = {Final Year Project Report},
  address      = {Sepang, Selangor, Malaysia},
  department   = {Department of Physics},
  note         = {Supervisor: Kelvin Ooi Jian Aun}
}
```

---

## License

This project is released under the **MIT License**. You are free to use, modify, and
distribute the code with attribution. See the accompanying `LICENSE` file for the full
text (or the MIT License terms at <https://opensource.org/licenses/MIT>).

© 2026 Tan Kok Jing · Xiamen University Malaysia
