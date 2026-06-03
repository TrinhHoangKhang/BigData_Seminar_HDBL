# MLFlow vs Weights & Biases Demo

Comparing **MLflow** and **Weights & Biases (W&B)** for experiment tracking on the same task: training a simple CNN on CIFAR-10 with PyTorch.

## Setup

```bash
pip install -r requirements.txt
wandb login
```

## Run W&B training

Single run (experiment tracking + wrong-prediction images + model artifact):

```bash
python train_wb.py
```

### W&B sweep (hyperparameter search)

```bash
wandb sweep sweep_config.yaml
wandb agent <SWEEP_ID>
```

Replace `<SWEEP_ID>` with the ID printed by `wandb sweep`.

## Run MLflow training

MLflow requires the tracking server to be running **before** training.

**Terminal 1** — start the UI:

```bash
mlflow ui --port 5000
```

**Terminal 2** — run training:

```bash
python train_mlflow.py
```

### Model serving 

```bash
mlflow models serve -m "models:/<Model_Registry>/<Model_alias>" -p 5001 --env-manager local
```
