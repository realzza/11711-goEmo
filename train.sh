export PYTHONPATH=$(pwd)

model_dir=goemotions/bert/cased_L-12_H-768_A-12/
python goemotions/bert_classifier.py \
    --multilabel true \
    --bert_config_file $model_dir/bert_config.json \
    --vocab_file $model_dir/vocab.txt \
    --output_dir exp