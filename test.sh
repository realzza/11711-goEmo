exp_dir=/ocean/projects/cis210027p/zzhou5/code/11711-goEmo/final_idea

for split in test valid; do
  # GOEMOTION TAXONOMY
  python test.py \
    --split=${split} \
    --db=bert-baseline-${split} \
    --checkpoint=${exp_dir}/final-bert-baseline/best_model.pt \
    --model-type=bert || exit 1

  python test.py \
    --split=${split} \
    --db=bert-emoji2vec-${split} \
    --use-emoji \
    --checkpoint=${exp_dir}/final-bert-emoji2vec/best_model.pt \
    --model-type=bert || exit 1

  python test.py \
    --split=${split} \
    --db=bert-emoji-text-align-${split} \
    --use-emoji \
    --text-align \
    --checkpoint=${exp_dir}/final-bert-emoji-text-align/best_model.pt \
    --model-type=bert || exit 1

  # EKMAN
  python test.py \
    --db=bert-ekman-${split} \
    --split=${split} \
    --task=ekman \
    --checkpoint=${exp_dir}/final-bert-ekman-base/best_model.pt \
    --model-type=bert || exit 1

  python test.py \
    --split=${split} \
    --db=bert-ekman-emoji2vec-${split} \
    --task=ekman \
    --use-emoji \
    --checkpoint=${exp_dir}/final-bert-ekman-emoji2vec/best_model.pt \
    --model-type=bert || exit 1

  python test.py \
    --split=${split} \
    --db=bert-ekman-emoji-text-align-${split} \
    --task=ekman \
    --use-emoji \
    --text-align \
    --checkpoint=${exp_dir}/final-bert-ekman-emoji-text-align/best_model.pt \
    --model-type=bert || exit 1

  # SENTIMENT
  python test.py \
    --db=bert-sentiment-${split} \
    --split=${split} \
    --task=sentiment \
    --checkpoint=${exp_dir}/final-bert-sentiment-base/best_model.pt \
    --model-type=bert || exit 1

  python test.py \
    --split=${split} \
    --db=bert-sentiment-emoji2vec-${split} \
    --task=sentiment \
    --use-emoji \
    --checkpoint=${exp_dir}/final-bert-sentiment-emoji2vec/best_model.pt \
    --model-type=bert || exit 1

  python test.py \
    --split=${split} \
    --db=bert-sentiment-emoji-text-align-${split} \
    --task=sentiment \
    --use-emoji \
    --text-align \
    --checkpoint=${exp_dir}/final-bert-sentiment-text-align/best_model.pt \
    --model-type=bert || exit 1
done
