python test.py \
  --db=pretrained-emoji-baseline-test \
  --use-emoji \
  --checkpoint=final_out/bert-cased-emoji2vec/best_model.pt \
  --model-type=bert || exit 1