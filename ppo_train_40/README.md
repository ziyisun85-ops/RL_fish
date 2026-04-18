# PPO Train 40 Upload Bundle

Run from the repo root with:

```powershell
python ppo_train_40/run.py
```

This folder is designed to be uploaded as one code bundle for the PPO-LoRA training entrypoint.

Required external resources:

- `runs/bc_pretrain/bc20_40_60_80_100_ppo/bc_large_pool_1_20__60_80_epoch020_actor.pth`
- `scenarios/large_pool_dataset_200/train/json`

Optional local fallbacks inside this folder:

- `ppo_train_40/weights/bc_large_pool_1_20__60_80_epoch020_actor.pth`
- `ppo_train_40/scenarios/train/json`

Outputs persist to:

- `runs/ppo_train_40/...`
