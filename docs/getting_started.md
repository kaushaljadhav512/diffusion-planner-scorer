# Getting started on a new GPU machine

Steps to take a fresh Ubuntu machine with an NVIDIA GPU to the point where it can run this repo's
simulations. Budget about 1–2 hours, mostly installs and downloads. The [README](../README.md) has more
detail on each step; this page is the checklist.

When done, continue with [running the week 1 evals](week1_evals.md).

## 1. Check the machine

- [ ] **Ubuntu 22.04**, native or WSL2: `lsb_release -a`
- [ ] **NVIDIA driver works:** `nvidia-smi` shows your GPU. For **RTX 50xx (Blackwell)** the driver must be
      **570 or newer**; the CUDA version in the top right of `nvidia-smi` must be ≥ 12.8. Older GPUs work
      with any recent driver.
- [ ] **RAM:** `free -g`. 16 GB minimum. Simulations run about one worker per 3 GB, so more RAM means faster
      runs.
- [ ] **Disk:** `df -h ~`. You need the size of the maps and test archives (shown on the nuPlan download
      page), twice over while the zips are still on disk, plus about 30 GB. If your home drive is too small,
      choose a bigger drive now; see step 4.

## 2. Get the repo

- [ ] Ask Kaushal for collaborator access to `kaushaljadhav512/diffusion-planner-scorer` (GitHub → Settings →
      Collaborators).
- [ ] Clone:
  ```bash
  sudo apt-get install -y git tmux
  git clone git@github.com:kaushaljadhav512/diffusion-planner-scorer.git   # or the https:// URL
  cd diffusion-planner-scorer
  ```

## 3. Install the environment (~30–45 min)

- [ ] `bash setup/bootstrap.sh`
  - It installs system packages (asks for `sudo` once), conda if missing, the `diffusion_planner` env
    (Python 3.9), nuplan-devkit and tuplan_garage at pinned commits, this repo, and the DP checkpoint.
  - **It picks the right PyTorch for your GPU automatically:** torch 2.7.1 / CUDA 12.8 on RTX 50xx, torch
    2.0.0 / CUDA 11.8 on anything else.
  - Safe to re-run after a failure.
  - **Shared machine** that already has a `diffusion_planner` env from someone else? Use your own:
    `CONDA_ENV=dps_<yourname> bash setup/bootstrap.sh`, and activate that name below.
- [ ] Check the last lines of the output:
  ```
  torch 2.7.1+cu128 | cuda available: True | NVIDIA GeForce RTX 5080     <- your GPU name / torch build
  cuda matmul ok: True                                                     <- must be True
  checkpoint loaded, top-level keys: ['model', 'ema_state_dict']
  ```
  If it says `cuda available: False` or the matmul fails, see Troubleshooting below.
- [ ] Open a **new shell**. Every shell from now on starts with:
  ```bash
  cd ~/diffusion-planner-scorer            # wherever you cloned it
  conda activate diffusion_planner         # or your CONDA_ENV name
  source setup/config.env
  ```
- [ ] `pip install pytest`. It's needed for the tests and isn't installed by setup.

## 4. Choose where the data goes

Data and simulation outputs go under `NUPLAN_HOME`, which defaults to `~/nuplan`. Keep the default if the
home drive has room. Otherwise, before downloading, point it at a bigger drive (ext4, and on WSL2 not under
`/mnt/c`) and make it permanent:
```bash
echo 'export NUPLAN_HOME=/path/to/big/drive/nuplan' >> ~/.bashrc && source ~/.bashrc
```
Every script reads this variable, so it has to be the same in every shell.

## 5. Download nuPlan (maps + test split)

The nuPlan license requires **each person to register and download with their own account**. Don't share
download links.

- [ ] Register at https://www.nuscenes.org/nuplan#download and accept the license.
- [ ] Get the two download URLs:
  1. On the download page, **left-click "US"** next to the archive. The link is generated when you click, so
     "copy link address" on the page only gives `nuplan#`.
  2. Open `chrome://downloads` (Ctrl+J), right-click the download's URL → **Copy link address**, then
     **cancel** the browser download.
  3. Do this for:
     - **nuPlan Maps** → "Zip containing Maps"
     - **nuPlan Test Split** → "Zip containing log databases for the test split"

     Not needed: mini, train, val, or anything labelled "Sensors".
  4. Each URL should end in a long signature (`?X-Amz-...`). It expires, so start the download soon.
- [ ] Put both URLs in `urls.txt` in the repo root, one per line. The file is gitignored.
- [ ] `bash setup/download_nuplan.sh urls.txt`. If it's interrupted, re-run it; downloads resume. If you get
      HTTP 403, copy fresh links.
- [ ] **Check** the `[layout]` summary at the end: a non-zero `test` DB count, and the map cities listed.
- [ ] Free the space: `rm $NUPLAN_HOME/downloads/*.zip`

## 6. Verify the machine (~20 min)

- [ ] `git pull`, to be on the latest code.
- [ ] `pytest tests/`. Everything should pass, **with no skips**; skips mean nuplan or the checkpoint wasn't
      found.
- [ ] `SMOKE=5 bash scripts/run_sim.sh dp test14-hard nr` simulates 5 scenarios end to end.
  - Its first line shows `threads=` (parallel simulations, RAM ÷ 3 GB by default) and `gpus=` (must be ≥ 1).
  - Note the wall-clock time; a 100-scenario run takes roughly that × 20.
- [ ] `python scripts/summarize_results.py` lists a `test14-hard_smoke5` row with a CLS score.

Note the GPU model, RAM, the `threads=` number and the smoke-test time; they are useful for estimating
how long longer runs will take.

## Notes

- **Several GPUs in one machine:** nothing to configure. `run_sim.sh` counts the GPUs and spreads the
  parallel simulations over them. The limit is usually RAM, not GPUs.
- **Long runs:** always inside `tmux new -s sim`. Detach with `Ctrl-b d`, reattach with `tmux attach -t sim`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cuda available: False` | `nvidia-smi` must work first (driver). On WSL2, install the NVIDIA driver on Windows only, never inside WSL. |
| `cuda matmul ok` fails, or "no kernel image is available" | Wrong torch build for the GPU. Check `nvidia-smi --query-gpu=compute_cap --format=csv`. On 12.x, update the driver to ≥ 570 and re-run `bootstrap.sh`. |
| `bootstrap.sh` fails installing nuplan requirements (omegaconf / hydra) | `git pull` (the fix pins `pip<24.1`), then re-run. |
| `download_nuplan.sh: usage: ... <file-with-one-url-per-line>` | Pass the URL file: `bash setup/download_nuplan.sh urls.txt`. |
| HTTP 403 while downloading | The links expired. Copy new ones and re-run. |
| Simulation killed, machine freezes, out of memory | Fewer parallel simulations: `THREADS=4 bash scripts/run_sim.sh ...` |
| `CUDA out of memory` | `GPU_FRAC=0.25 THREADS=4 bash scripts/run_sim.sh ...` |
| Zero or too few scenarios found | Test DBs must be in `$NUPLAN_DATA_ROOT/nuplan-v1.1/test/*.db`, and `NUPLAN_HOME` must match the one used for the download. |
| Hydra error `could not find ... scenario_filter` | Env not active or install incomplete: `conda activate diffusion_planner`, re-run `bootstrap.sh`. |
