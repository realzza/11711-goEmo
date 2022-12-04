python train.py --db base-baseline --sweep-count 12
python train.py --db pretrained-emoji-baseline --use-emoji --sweep-count 12
python train.py --db emoji-unk-baseline --use-emoji --sweep-count 12
python train.py --db emoji-rand-baseline --use-emoji --sweep-count 12 --emoji-rand-init