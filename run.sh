python train.py --model-type bert --db final-bert-ekman-base --sweep-count 4 --logdir final_idea/ --task ekman
python train.py --model-type bert --db final-bert-ekman-emoji2vec --use-emoji --sweep-count 4 --logdir final_idea/ --task ekman
python train.py --model-type bert --db final-bert-ekman-emoji-text-align --use-emoji --text-align --sweep-count 4 --logdir final_idea/ --task ekman
python train.py --model-type bert --db final-bert-sentiment-base --sweep-count 4 --logdir final_idea/ --task sentiment
python train.py --model-type bert --db final-bert-sentiment-emoji2vec --use-emoji --sweep-count 4 --logdir final_idea/ --task sentiment
python train.py --model-type bert --db final-bert-sentiment-text-align --text-align --use-emoji --sweep-count 4 --logdir final_idea/ --task sentiment
# python train.py --model-type roberta --db roberta-sentiment --sweep-count 4 --logdir final_out/ --task sentiment
# python train.py --model-type roberta --db roberta-ekman --sweep-count 2 --logdir final_out/ --task ekman
# python train.py --model-type roberta --db roberta-sentiment-emoji2vec --use-emoji --sweep-count 2 --logdir final_out/ --task sentiment
# python train.py --model-type roberta --db roberta-ekman-emoji2vec --use-emoji --sweep-count 2 --logdir final_out/ --task ekman

python train.py --model-type bert --db final-bert-baseline --sweep-count 4 --logdir final_idea/
python train.py --model-type bert --db final-bert-emoji2vec --use-emoji --sweep-count 4 --logdir final_idea/
python train.py --model-type bert --db final-bert-emoji-text-align --use-emoji --text-align --sweep-count 4 --emoji-rand-init --logdir final_idea/
# python train.py --model-type roberta --db roberta-baseline --sweep-count 8 --logdir final_out/
# python train.py --model-type roberta --db roberta-emoji2vec --use-emoji --sweep-count 8 --logdir final_out/