# PPO Train 1 40 60 100 Upload Bundle

Run from the repo root with:

```powershell
python ppo_train_1_40_60_100/run.py
```

This folder is designed to be uploaded as one code bundle for the PPO-LoRA training entrypoint.

Required external resources:

- `runs/bc_pretrain/bc20_40_60_80_100_ppo/bc_large_pool_1_40__60_100_epoch020_actor.pth`
- `scenarios/large_pool_dataset_200/train/json`

Optional local fallbacks inside this folder:

- `ppo_train_1_40_60_100/weights/bc_large_pool_1_40__60_100_epoch020_actor.pth`
- `ppo_train_1_40_60_100/scenarios/train/json`

Outputs persist to:

- `runs/ppo_train_1_40_60_100/...`
