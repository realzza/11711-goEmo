# goEmotions w/ PyTorch

## Setup
```
conda create -n goemo python=3.9
pip install -r requirement.txt
```

## Train
```
python train.py
```
Checkout runtime visualizations in wandb links generated in training process.

## Results 
Visualized using [wandb](https://github.com/wandb/wandb)

- goemo-baseline: https://wandb.ai/realzza/goemo-baseline
- goemo-emoji-default-init: https://wandb.ai/realzza/goemo-emoji-default-init
- goemo-emoji-[e2v](e2vDemo/emoji_torch.ipynb)-init: TODO

## Todo
- [ ] add inference script for models
- [ ] add sweep id for each run in wandb