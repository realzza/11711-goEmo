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
- goemo-emoji-[e2v](e2vDemo/emoji_torch.ipynb)-init: ~~TODO~~ DONE

### 8-sweep Experiments
- base-baseline: https://wandb.ai/realzza/base-baseline
- pretrained-emoji-baseline: https://wandb.ai/realzza/pretrained-emoji-baseline
- emoji-unk-baseline: https://wandb.ai/realzza/emoji-unk-baseline
- emoji-rand-baseline: https://wandb.ai/realzza/emoji-rand-baseline

## Todo
- [ ] add inference script for models 
- [ ] add sweep id for each run in wandb
- [ ] add valid log for emoji2vec train (@tjysdsg)
- [ ] add emoticon -> description (@tjysdsg)