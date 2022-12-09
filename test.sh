# # GOEMOTION TAXONOMY
# python test.py \
#   --db=bert-baseline-test \
#   --checkpoint=final_out/bert-cased-baseline/best_model.pt \
#   --model-type=bert || exit 1
#
# python test.py \
#   --db=bert-emoji2vec-test \
#   --use-emoji \
#   --checkpoint=final_out/bert-cased-emoji2vec/best_model.pt \
#   --model-type=bert || exit 1

python test.py \
  --db=roberta-test \
  --checkpoint=final_out/roberta-baseline/best_model.pt \
  --model-type=roberta || exit 1

# TODO
#  python test.py \
#    --db=roberta-emoji2vec-test \
#    --use-emoji \
#    --checkpoint=final_out/roberta-emoji2vec/best_model.pt \
#    --model-type=roberta || exit 1

# TODO
#  python test.py \
#    --db=bert-emoji-rand-test \
#    --use-emoji \
#    --emoji-rand-init \
#    --checkpoint=final_out/bert-cased-emoji-rand/best_model.pt \
#    --model-type=bert || exit 1

# EKMAN
# python test.py \
#   --db=bert-ekman-test \
#   --task=ekman \
#   --checkpoint=final_out/bert-ekman/best_model.pt \
#   --model-type=bert || exit 1
#
# python test.py \
#   --db=bert-ekman-emoji2vec-test \
#   --task=ekman \
#   --use-emoji \
#   --checkpoint=final_out/bert-ekman-emoji2vec/best_model.pt \
#   --model-type=bert || exit 1

python test.py \
  --db=roberta-ekman-test \
  --task=ekman \
  --checkpoint=final_out/roberta-ekman/best_model.pt \
  --model-type=roberta || exit 1

python test.py \
  --db=roberta-ekman-emoji2vec-test \
  --task=ekman \
  --use-emoji \
  --checkpoint=final_out/roberta-ekman-emoji2vec/best_model.pt \
  --model-type=roberta || exit 1
