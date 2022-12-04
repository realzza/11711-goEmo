python train.py --db base-baseline --sweep-count 8
python train.py --db pretrained-emoji-baseline --use-emoji --sweep-count 8
python train.py --db emoji-unk-baseline --use-emoji --sweep-count 8
python train.py --db emoji-rand-baseline --use-emoji --sweep-count 8 --emoji-rand-init