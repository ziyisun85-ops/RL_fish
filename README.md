# RL_fish

MuJoCo-based robotic fish project with BC, PPO, and SAC training pipelines.

This README is written as a working map of the current repository. It focuses on:

- project architecture
- main training and evaluation entry points
- scenario and dataset layout
- where each iteration's results are stored
- which initial weights each PPO or SAC run starts from
- where resumed checkpoints and analysis files are written

## 1. Project structure

The current repo is organized around the simulation environment, scenario datasets, BC training, RL training, and result analysis.

| Path | Role |
| --- | --- |
| `configs/default_config.py` | Default environment, reward, PPO/SAC, and evaluation configuration |
| `envs/fish_env.py` | Main Gymnasium environment, including reset, step, reward, termination, rendering, and scenario loading |
| `hydrodynamics.py` | Hydrodynamic force application during MuJoCo stepping |
| `algorithms/episode_cycle_ppo.py` | Custom PPO variant for episode-cycle / scenario-cycle updates |
| `utils/lora.py` | LoRA module implementation |
| `utils/lora_policy.py` | LoRA-enabled PPO policy |
| `utils/policy_utils.py` | Env/model builders, weight loading helpers, actor-state loading |
| `utils/scenario_io.py` | Fixed-scenario JSON read/write and dataset-manifest env override loading |
| `utils/obstacles.py` | Obstacle generation and local obstacle observation |
| `model/` | MuJoCo XML scene files, fish model, pool model, STL assets, XML generation helpers |
| `scenarios/` | Fixed scenario JSON datasets for BC, PPO, and evaluation |
| `runs/` | All generated artifacts: demos, BC checkpoints, PPO checkpoints, logs, summaries, plots, eval outputs |
| `train.py` | Main PPO/SAC training entry point |
| `train_bc.py` | Behavior cloning training entry point |
| `collect_bc_data.py` | Human demonstration collection |
| `evaluate_bc_rl.py` | Shared evaluation script for BC weights and PPO `.zip` models |
| `train_curriculum.py` | Sequential scenario curriculum training launcher |
| `generate_training_envs.py` | Generate older small fixed training scenarios |
| `generate_large_pool_dataset.py` | Generate the large-pool 200-scenario dataset |
| `plot_success_scatter.py` | Plot episode-level success/failure scatter |
| `plot_success_by_weight.py` | Plot checkpoint-level success rate by weight name |
| `reward_function.md` | Standalone reward-function explanation |
| `add_hydrodynamic.md` | Hydrodynamic modeling notes |

## 2. Runtime architecture

The main training loop is:

1. `train.py` builds the environment with `utils/policy_utils.py`.
2. `envs/fish_env.py` loads MuJoCo XML from `model/fish_pool_scene.xml`.
3. Each step collects:
   - head camera image
   - IMU-like signals
   - relative goal information
4. PPO, SAC, or BC policy outputs the high-level steering command.
5. `utils/mappings.py` maps the steering command into head motion and tail-frequency control.
6. `hydrodynamics.py` applies fluid forces before each MuJoCo substep.
7. `envs/fish_env.py` computes reward, success, timeout, collision, and logging fields.
8. Training outputs are written under `runs/`.

The core current training variants are:

- BC on demonstration datasets
- PPO from BC actor initialization
- PPO scenario-cycle training on fixed 20 scenes
- PPO random-20-per-update training from a previous scenario-cycle checkpoint
- SAC from BC actor initialization with PPO-to-SAC actor weight mapping
- optional LoRA fine-tuning for PPO

`train.py` now supports `--algo sac` in addition to the existing PPO path. The SAC branch keeps the same environment pipeline, scenario-cycle logging, and `update_000001` checkpoint naming. When `--bc-weights` points to a BC checkpoint produced by `train_bc.py`, the loader maps PPO actor weights into the SAC actor as follows:

- `features_extractor.*` -> `actor.features_extractor.*`
- `mlp_extractor.policy_net.*` -> `actor.latent_pi.*`
- `action_net.*` -> `actor.mu.*`
- PPO `log_std` initializes `actor.log_std.bias`, with `actor.log_std.weight` zeroed so the initial SAC std is state-independent

## 3. Scenario and dataset layout

There are two main scenario families in this repo.

### 3.1 Older small-pool fixed scenarios

| Path | Notes |
| --- | --- |
| `scenarios/training_envs/` | Older 20-scene small-pool training set |
| `scenarios/bc_demo_envs/` | Small set of BC demo scenarios |

These are mostly the earlier fixed-scenario experiments.

### 3.2 Current large-pool dataset

| Path | Notes |
| --- | --- |
| `scenarios/large_pool_dataset_200/dataset_manifest.json` | Dataset-level config |
| `scenarios/large_pool_dataset_200/train/json/` | 100 training scenarios: `train_env_001.json` to `train_env_100.json` |
| `scenarios/large_pool_dataset_200/test/json/` | 100 test scenarios: `test_env_001.json` to `test_env_100.json` |
| `scenarios/large_pool_dataset_200/train/topdown/` | Top-down PNG preview for train scenarios |
| `scenarios/large_pool_dataset_200/test/topdown/` | Top-down PNG preview for test scenarios |

This is the current main dataset for BC and PPO.

## 4. `runs/` directory map

This is the most important section if you want to quickly find outputs.

| Path | What is stored there |
| --- | --- |
| `runs/bc_demos/` | Older BC demonstration datasets |
| `runs/bc_demos_large_pool_100/` | Large-pool BC demonstration datasets collected scene-by-scene |
| `runs/bc_pretrain/` | BC checkpoints and metrics |
| `runs/ppo_fish_baseline/` | PPO runs, checkpoints, episode logs, summaries, plots |
| `runs/eval/` | BC/PPO evaluation JSON files and evaluation subfolders |

If you are unsure what a run started from, the authoritative file is the run config:

- `config.json`
- or `config_<run_id>.json`

The fields that matter most are:

- `bc_weights`
- `resume_from`
- `resume_policy_weights`
- `selected_scenario_cycle_paths`
- `existing_episode_count_at_start`
- `existing_cycle_update_index_at_start`

## 5. BC results and where they are stored

### 5.1 Older small-scenario BC

| Run | Output path | Main weights |
| --- | --- | --- |
| Training on older `training_env_01` demos | `runs/bc_pretrain/training_env_01/` | `bc_fish_env01.zip`, `bc_fish_env01_actor.pth` |
| 5-demo restart experiment | `runs/bc_pretrain/five_episodes/` | `bc_5episodes_restart.zip`, `bc_5episodes_restart_actor.pth` |

These runs also store metrics JSON files such as:

- `runs/bc_pretrain/training_env_01/bc_fish_env01_metrics.json`
- `runs/bc_pretrain/five_episodes/bc_5episodes_restart_metrics.json`

### 5.2 Current large-pool BC

Current large-pool BC checkpoints are under:

- `runs/bc_pretrain/large_pool_bc_v2/checkpoints/`

Available saved epochs currently include:

- `bc_large_pool_v2_epoch010_actor.pth`
- `bc_large_pool_v2_epoch020_actor.pth`
- `bc_large_pool_v2_epoch030_actor.pth`
- `bc_large_pool_v2_epoch040_actor.pth`

Full-model `.zip` files are stored next to them in the same directory.

There is also a packaged archive:

- `runs/bc_pretrain/large_pool_bc_v2/bc_checkpoints.zip`

### 5.3 Current BC data source

The current large-pool BC training data is primarily under:

- `runs/bc_demos_large_pool_100/`

Older demo data is under:

- `runs/bc_demos/`

### 5.4 BC weight most commonly used as PPO initialization

The main downstream PPO runs in this repo start from:

- `runs/bc_pretrain/large_pool_bc_v2/checkpoints/bc_large_pool_v2_epoch020_actor.pth`

This is the most important BC actor checkpoint to keep track of.

## 6. PPO results by training variant

### 6.1 Older fixed-scenario PPO baseline

This is an older small-pool fixed-scenario run, not the current large-pool BC-initialized pipeline.

Output directory:

- `runs/ppo_fish_baseline/training_env_01/`

Main files:

- `config.json`
- `episode_metrics.csv`
- `monitor.csv`
- `ppo_fish_baseline.zip`
- `ppo_fish_baseline_policy.pth`
- `ppo_fish_baseline_interrupted.zip`
- `ppo_fish_baseline_interrupted_policy.pth`
- `checkpoints/`
- `videos/`

Scenario used:

- `scenarios/training_envs/training_env_01.json`

This run is mainly useful as an older baseline reference.

### 6.2 BC-initialized PPO on fixed large-pool scenes

The fixed-scene large-pool PPO chain is currently represented by:

- `runs/ppo_fish_baseline/train_env_001/`
- `runs/ppo_fish_baseline/train_env_002/`

The first fixed-scene run starts from BC actor:

- `runs/bc_pretrain/large_pool_bc_v2/checkpoints/bc_large_pool_v2_epoch020_actor.pth`

This is recorded in:

- `runs/ppo_fish_baseline/train_env_001/config_BC_RL_train100_20260414_train_env_001.json`

Then the run is resumed from its own interrupted PPO checkpoints:

- `runs/ppo_fish_baseline/train_env_001/ppo_fish_baseline_interrupted.zip`
- `runs/ppo_fish_baseline/train_env_001/ppo_fish_baseline_interrupted_BC_RL_train100_20260414_train_env_001.zip`

The follow-up `train_env_002` run resumes from the last interrupted checkpoint of `train_env_001`:

- `runs/ppo_fish_baseline/train_env_001/ppo_fish_baseline_interrupted_BC_RL_train100_20260414_resume2_train_env_001.zip`

Checkpoint storage:

- `runs/ppo_fish_baseline/train_env_001/checkpoints/`
- naming rule: `ppo_fish_baseline_episode_000020.zip` and `ppo_fish_baseline_episode_000020_policy.pth`

This run currently has episode-based checkpointing and already includes many saved episodes up to the high hundreds.

Per-run outputs in the same directory:

- `episode_metrics.csv`
- `monitor_*.monitor.csv`
- `training_summary*.json`
- `videos/`

### 6.3 BC-initialized PPO on fixed 20 training scenes

This is the main fixed-20-scene scenario-cycle run.

Output directory:

- `runs/ppo_fish_baseline/scenario_cycle/`

Initial weight source:

- `runs/bc_pretrain/large_pool_bc_v2/checkpoints/bc_large_pool_v2_epoch020_actor.pth`

This is recorded in:

- `runs/ppo_fish_baseline/scenario_cycle/config_lora_cycle_train20_update20_save20_20260414_restart1.json`

Selected scenes:

- `scenarios/large_pool_dataset_200/train/json/train_env_001.json`
- through `scenarios/large_pool_dataset_200/train/json/train_env_020.json`

Resume chain:

- first restart from BC actor
- later resume from `checkpoints/ppo_fish_baseline_update_000004.zip`
- later resume from `checkpoints/ppo_fish_baseline_update_000035.zip`

Checkpoint storage:

- `runs/ppo_fish_baseline/scenario_cycle/checkpoints/`

Checkpoint naming rule:

- full model: `ppo_fish_baseline_update_000001.zip`
- policy-only snapshot: `ppo_fish_baseline_update_000001_policy.pth`

Current known checkpoint range in this directory:

- starts at `update_000001`
- currently goes at least to `update_000063`

Important analysis files in the same run directory:

- `episode_metrics.csv`
- `training_summary.json`
- `success_by_weight.csv`
- `success_trend_every20.csv`
- `success_rate_by_weight_name.png`
- `success_rate_by_weight_stem.png`

### 6.4 Random-20-per-update PPO from the 100-scene training pool

This is the random-20 scenario-cycle run.

Output directory:

- `runs/ppo_fish_baseline/scenario_cycle_rand20_from_update25_20260415/scenario_cycle/`

Scenario pool:

- `scenarios/large_pool_dataset_200/train/json/train_env_001.json`
- through `scenarios/large_pool_dataset_200/train/json/train_env_100.json`

Sampling rule:

- each PPO update samples 20 unique scenes from the 100-scene pool
- one episode per sampled scene per update

Initial weight source for this run:

- `runs/ppo_fish_baseline/scenario_cycle/checkpoints/ppo_fish_baseline_update_000025_policy.pth`

This is recorded in:

- `runs/ppo_fish_baseline/scenario_cycle_rand20_from_update25_20260415/scenario_cycle/config.json`

Checkpoint storage:

- `runs/ppo_fish_baseline/scenario_cycle_rand20_from_update25_20260415/scenario_cycle/checkpoints/`

Checkpoint naming rule:

- full model: `ppo_fish_rand20_from_u025_update_000026.zip`
- policy-only snapshot: `ppo_fish_rand20_from_u025_update_000026_policy.pth`

Current known checkpoint range in this directory:

- starts at `update_000026`
- currently goes at least to `update_000049`

Important analysis files in the same run directory:

- `episode_metrics.csv`
- `training_summary.json`
- `train_success_by_weight_all.csv`
- `train_success_by_weight_all.png`
- `success_rate_by_weight_name.png`
- `success_rate_by_weight_stem.png`

## 7. Where evaluation results are stored

Evaluation outputs are written under:

- `runs/eval/`

This directory contains:

- BC evaluation JSON files
- PPO evaluation JSON files
- per-scenario evaluation subdirectories

Examples already present:

- `runs/eval/bc_large_pool_v2_epoch020_test_env010.json`
- `runs/eval/bc_large_pool_v2_epoch020_train_env001.json`
- `runs/eval/ppo_rand20_u037_test001_020_ep5_cuda/`

## 8. Weight naming rules

This is useful when tracing lineage between runs.

### 8.1 BC checkpoints

- `*_epoch010.zip`
- `*_epoch010_actor.pth`

Meaning:

- `.zip` is the full SB3 model package
- `_actor.pth` is the actor-state checkpoint usually used to initialize PPO

### 8.2 Fixed-scene PPO checkpoints

- `ppo_fish_baseline_episode_000020.zip`
- `ppo_fish_baseline_episode_000020_policy.pth`

Meaning:

- checkpoint is keyed by completed episode count

### 8.3 Scenario-cycle PPO checkpoints

- `ppo_fish_baseline_update_000001.zip`
- `ppo_fish_baseline_update_000001_policy.pth`

Meaning:

- checkpoint is keyed by PPO update index, not raw episode count

### 8.4 Random-20 scenario-cycle checkpoints

- `ppo_fish_rand20_from_u025_update_000026.zip`
- `ppo_fish_rand20_from_u025_update_000026_policy.pth`

Meaning:

- this run is a separate branch initialized from scenario-cycle `update_000025`

### 8.5 Interrupted snapshots

Files containing `interrupted` are recovery snapshots saved when a run is stopped or interrupted.

Examples:

- `ppo_fish_baseline_interrupted.zip`
- `ppo_fish_baseline_interrupted_policy.pth`
- `ppo_fish_rand20_from_u025_interrupted.zip`

Use these for resume/recovery, not as the primary numbered checkpoint history.

## 9. Recommended artifact lookup order

If you want the current main large-pool training chain, follow artifacts in this order:

1. BC actor:
   `runs/bc_pretrain/large_pool_bc_v2/checkpoints/bc_large_pool_v2_epoch020_actor.pth`
2. Fixed 20-scene PPO branch:
   `runs/ppo_fish_baseline/scenario_cycle/`
3. Random-20-from-100 PPO branch:
   `runs/ppo_fish_baseline/scenario_cycle_rand20_from_update25_20260415/scenario_cycle/`

If you want the single-scene BC-to-PPO chain, follow:

1. `runs/bc_pretrain/large_pool_bc_v2/checkpoints/bc_large_pool_v2_epoch020_actor.pth`
2. `runs/ppo_fish_baseline/train_env_001/`
3. `runs/ppo_fish_baseline/train_env_002/`

## 10. Quick reference

### 10.1 Current most important initialization weights

| Use case | Weight path |
| --- | --- |
| BC actor used to start large-pool PPO | `runs/bc_pretrain/large_pool_bc_v2/checkpoints/bc_large_pool_v2_epoch020_actor.pth` |
| Fixed 20-scene PPO branch for later random20 branch | `runs/ppo_fish_baseline/scenario_cycle/checkpoints/ppo_fish_baseline_update_000025_policy.pth` |

### 10.2 Current most important result directories

| Stage | Directory |
| --- | --- |
| BC demos | `runs/bc_demos_large_pool_100/` |
| BC checkpoints | `runs/bc_pretrain/large_pool_bc_v2/checkpoints/` |
| Fixed-scene BC-to-PPO | `runs/ppo_fish_baseline/train_env_001/` |
| Fixed 20-scene PPO | `runs/ppo_fish_baseline/scenario_cycle/` |
| Random-20 PPO from 100-scene pool | `runs/ppo_fish_baseline/scenario_cycle_rand20_from_update25_20260415/scenario_cycle/` |
| Evaluation outputs | `runs/eval/` |

## 11. Notes

- `README.md` is now intended to be a runbook for locating artifacts, not only a conceptual overview.
- If a path in `runs/` has multiple `config_*.json` files, read those files first before reusing a checkpoint.
- For downstream scripting, prefer treating `config.json` and `training_summary.json` as the source of truth for each run.
