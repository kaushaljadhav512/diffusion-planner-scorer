# Learning to Pick the Plan: A Learned Trajectory Scorer for Diffusion Planner

CMU 11-785 course project. The project proposal is in [`docs/IDL_Project_proposal.pdf`](docs/IDL_Project_proposal.pdf).

This repository is built on [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) (Zheng et al., ICLR 2025), starting from upstream commit `a3a621f`. The upstream README is kept as [`UPSTREAM_README.md`](UPSTREAM_README.md). All credit for the base planner belongs to its authors; this repository is private and for coursework only.

This document describes how to set up the nuPlan workspace on a local machine and reproduce the midterm baseline: the released Diffusion Planner checkpoint, evaluated in closed-loop simulation on the Test14-hard benchmark with non-reactive (NR) and reactive (R) traffic agents.

---

## 0. Requirements

| Requirement | Minimum | Verify with |
|---|---|---|
| Operating system | Ubuntu 22.04, or Windows 11 with WSL2 (Ubuntu 22.04). macOS is not supported. | `lsb_release -a` |
| Memory | 16 GB (32 GB roughly halves simulation time) | `free -g` |
| Free disk space | Size of the maps and test-split archives (listed on the nuPlan download page) plus 30 GB | `df -h ~` |
| GPU | Optional: NVIDIA GPU with ≥ 4 GB memory. Without one, the planner runs on CPU at reduced speed. | `nvidia-smi` |

**WSL2:** store the dataset inside the Linux filesystem (e.g. under `~`), not on a Windows-mounted drive (`/mnt/c`, `/mnt/d`). nuPlan reads its SQLite databases continuously, and access through Windows mounts is an order of magnitude slower. If the system drive lacks space, [relocate the WSL virtual disk](https://learn.microsoft.com/en-us/windows/wsl/disk-space). For GPU support, install the standard Windows NVIDIA driver only; no driver should be installed inside the WSL distribution.

**External storage (native Ubuntu):** if internal disk space is insufficient, use an ext4-formatted external SSD and set the data location in every shell before running any command below:
```bash
export NUPLAN_HOME=/media/$USER/<drive-name>/nuplan
```
If `NUPLAN_HOME` is not set, data is stored under `~/nuplan`.

---

## 1. Clone the repository

```bash
sudo apt-get install -y git tmux
git clone git@github.com:kaushaljadhav512/diffusion-planner-scorer.git
cd diffusion-planner-scorer
```
Repository access must be granted by the owner, under Settings → Collaborators. If no SSH key is configured with GitHub, clone over HTTPS instead: `https://github.com/kaushaljadhav512/diffusion-planner-scorer.git`.

## 2. Install the environment (approx. 30–45 min)

```bash
bash setup/bootstrap.sh
```
The script performs the following steps:
- Installs required system packages (prompts for `sudo` once).
- Installs Miniforge (conda) if conda is not already present.
- Creates the `diffusion_planner` conda environment (Python 3.9).
- Installs nuplan-devkit and tuplan_garage at the commits pinned in `setup/config.env`, and this repository (the `diffusion_planner` package) in editable mode.
- Downloads the Diffusion Planner checkpoint.
- Writes the resolved package versions to `setup/pip-freeze.txt`.

The script is idempotent and can be re-run after a failure. On completion it reports whether CUDA is available and confirms that the checkpoint loads.

Open a new shell after installation. Every subsequent shell requires:
```bash
cd diffusion-planner-scorer
conda activate diffusion_planner
source setup/config.env          # export NUPLAN_HOME first if using external storage
```

## 3. Download the nuPlan data

nuPlan is distributed under Motional's dataset license. **Each user must register individually and accept the license.** Download links are personal and must not be shared.

1. Register at <https://www.nuscenes.org/nuplan#download> and accept the license terms.
2. From the download page, copy the links for these two archives only:
   - **Maps:** `nuplan-maps-v1.1` (or the latest maps release)
   - **Test split:** the nuPlan v1.1 **test** database archive. The train, val and sensor-blob archives are not needed.
3. Save the links to `urls.txt` in the repository root, one per line. This file is excluded from version control. The links expire, so start the download promptly.
4. Run:
   ```bash
   bash setup/download_nuplan.sh urls.txt
   ```
   Interrupted downloads resume when the script is re-run. On completion, the script prints the number of databases per split, and the `test` count must be non-zero. The archives in `$NUPLAN_HOME/downloads` can then be deleted.

## 4. Smoke test (approx. 5–15 min)

```bash
SMOKE=5 bash scripts/run_sim.sh dp test14-hard nr
```
This simulates the first five scenarios and verifies the pipeline end to end.
- The first output line reports the number of parallel simulations (`threads`), which defaults to available RAM ÷ 3 GB, and the number of GPUs detected.
- Record the run time. A full run takes approximately (smoke-test time) × 272 / 5.
- `python scripts/summarize_results.py` should list a `test14-hard_smoke5` row with a closed-loop score (CLS).

## 5. Baseline runs

Each full run covers 272 scenarios and can take several hours on a laptop. Run inside `tmux` so the job continues if the terminal closes:
```bash
tmux new -s sim
bash scripts/run_sim.sh dp test14-hard nr      # Run 1: non-reactive agents
bash scripts/run_sim.sh dp test14-hard r       # Run 2: reactive agents
```
Detach with `Ctrl-b d`, and reattach with `tmux attach -t sim`.

Alternatively, run both configurations in sequence, with per-run log files and a summary at the end:
```bash
PLANNERS=dp SPLITS=test14-hard bash scripts/run_baselines.sh
```

## 6. Report results

```bash
python scripts/summarize_results.py --by-type --csv results/dp_test14hard.csv | tee results/dp_test14hard.txt
git add results/ && git commit -m "Diffusion Planner Test14-hard baseline (NR + R)" && git push
```
The summary lists each score alongside the published values (75.99 NR, 69.22 R) and the difference. Also record the hardware used (CPU, RAM, GPU) and the wall-clock time of each run.

---

## Troubleshooting

| Symptom | Resolution |
|---|---|
| System becomes unresponsive, process killed, or out-of-memory errors | Reduce parallelism: `THREADS=2 bash scripts/run_sim.sh ...` |
| `CUDA out of memory` | `GPU_FRAC=0.5 THREADS=2 bash scripts/run_sim.sh ...`, or force CPU execution with `CUDA_VISIBLE_DEVICES= bash scripts/run_sim.sh ...` |
| Fewer than 272 scenarios, or zero scenarios found | Test databases are missing or misplaced. They must be in `$NUPLAN_DATA_ROOT/nuplan-v1.1/test/*.db`, and `NUPLAN_HOME` must match the location used for the download. |
| Hydra errors such as `could not find ... scenario_filter` | The environment is inactive or the installation is incomplete. Activate `diffusion_planner` and re-run `bash setup/bootstrap.sh`. |
| Import errors involving `ray` or `numpy` | Report the full error together with `setup/pip-freeze.txt`. |
| HTTP 403 on download | The links have expired. Copy new links from the nuPlan download page. |

## Repository layout

```
diffusion_planner/            Diffusion Planner package (upstream code; project extensions are added here)
checkpoints/                  released Diffusion Planner checkpoint (downloaded by bootstrap; not version-controlled)
setup/config.env              paths and pinned upstream commits (override NUPLAN_HOME via export)
setup/bootstrap.sh            one-time environment installation
setup/download_nuplan.sh      download and arrange nuPlan archives
scripts/run_sim.sh            single closed-loop simulation (planner, split, nr|r)
scripts/run_baselines.sh      grid of simulations followed by a summary
scripts/summarize_results.py  results table with comparison against published scores
scripts/plot_baseline.py      per-scenario-type CLS figure for the report
docs/aws.md                   notes for running on AWS
results/                      committed result tables
report/                       midterm/final report sources and figures
third_party/                  nuplan-devkit and tuplan_garage (created by bootstrap; not version-controlled)
```

## Known limitation: Diffusion Planner with refinement

The paper's rule-based refinement variant is not included in the public release. The authors used the refinement module from State Transformer 2 ([Diffusion-Planner#14](https://github.com/ZhengYinan-AIR/Diffusion-Planner/issues/14)). The midterm baseline therefore covers only the model without refinement. The rule-based comparison baseline will either port the STR2 refinement module or select among sampled candidates using the PDM-Closed scorer; this choice has not yet been made.
