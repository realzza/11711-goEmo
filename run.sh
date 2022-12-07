python train.py --db base-baseline-log --sweep-count 1 --logdir out/
python train.py --db pretrained-emoji-baseline-log --use-emoji --sweep-count 1 --logdir out/
python train.py --db emoji-unk-baseline-log --use-emoji --sweep-count 1 --logdir out/
python train.py --db emoji-rand-baseline-log --use-emoji --sweep-count 1 --emoji-rand-init --logdir out/