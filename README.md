# Quickstart

## Setup

Install python 3.7 tensorflow 1.15 cudnn 7.6.5 numpy 1.19

Then

```bash
pip install -r goemotions/requirements.txt
```

```bash
cd goemotions
./setup.sh
cd ..
```

## Train

```bash
./train.sh
```

# Baseline Results

```bash
model_dir=goemotions/bert/cased_L-12_H-768_A-12/
python goemotions/bert_classifier.py \
    --multilabel true \
    --bert_config_file $model_dir/bert_config.json \
    --vocab_file $model_dir/vocab.txt \
    --output_dir exp
```

    F1_at_threshold_0.00 = 0.07996891
    F1_at_threshold_0.10 = 0.46654633
    F1_at_threshold_0.20 = 0.53667164
    F1_at_threshold_0.30 = 0.5466584
    F1_at_threshold_0.40 = 0.5287117
    F1_at_threshold_0.50 = 0.4981706
    F1_at_threshold_0.60 = 0.44387728
    F1_at_threshold_0.70 = 0.38723078
    F1_at_threshold_0.80 = 0.29982966
    F1_at_threshold_0.90 = 0.1547105
    F1_at_threshold_0.95 = 0.06891227
    F1_at_threshold_0.99 = 0.0
    accuracy = 0.83669496
    accuracy_weighted = 0.8488815
    admiration_accuracy = 0.9303483
    admiration_auc = 0.9076752
    admiration_precision = 0.6334746
    admiration_recall = 0.59325397
    amusement_accuracy = 0.97512436
    amusement_auc = 0.970537
    amusement_precision = 0.70347005
    amusement_recall = 0.844697
    anger_accuracy = 0.9585406
    anger_auc = 0.8551353
    anger_precision = 0.4262295
    anger_recall = 0.3939394
    annoyance_accuracy = 0.92905843
    annoyance_auc = 0.7759436
    annoyance_precision = 0.32804233
    annoyance_recall = 0.19375
    approval_accuracy = 0.9235305
    approval_auc = 0.75991005
    approval_precision = 0.3504673
    approval_recall = 0.21367522
    auc = 0.83669496
    auc_weighted = 0.8488815
    caring_accuracy = 0.96959645
    caring_auc = 0.81646436
    caring_precision = 0.2972973
    caring_recall = 0.16296296
    confusion_accuracy = 0.96885943
    confusion_auc = 0.8961113
    confusion_precision = 0.4047619
    confusion_recall = 0.22222222
    curiosity_accuracy = 0.9449051
    curiosity_auc = 0.9256843
    curiosity_precision = 0.48021108
    curiosity_recall = 0.64084506
    desire_accuracy = 0.98489034
    desire_auc = 0.7999039
    desire_precision = 0.51428574
    desire_recall = 0.21686748
    disappointment_accuracy = 0.97217613
    disappointment_auc = 0.74654996
    disappointment_precision = 0.5
    disappointment_recall = 0.0066225166
    disapproval_accuracy = 0.9384559
    disapproval_auc = 0.80379677
    disapproval_precision = 0.29192546
    disapproval_recall = 0.17602997
    disgust_accuracy = 0.9784411
    disgust_auc = 0.89236915
    disgust_precision = 0.5405405
    disgust_recall = 0.32520324
    embarrassment_accuracy = 0.99318224
    embarrassment_auc = 0.8021988
    embarrassment_precision = 0.0
    embarrassment_recall = 0.0
    excitement_accuracy = 0.9810208
    excitement_auc = 0.8848264
    excitement_precision = 0.5
    excitement_recall = 0.2815534
    fear_accuracy = 0.9887599
    fear_auc = 0.9123201
    fear_precision = 0.6976744
    fear_recall = 0.3846154
    global_step = 10852
    gratitude_accuracy = 0.98968124
    gratitude_auc = 0.9922787
    gratitude_precision = 0.92774564
    gratitude_recall = 0.9119318
    grief_accuracy = 0.9988944
    grief_auc = 0.7905215
    grief_precision = 0.0
    grief_recall = 0.0
    joy_accuracy = 0.9747558
    joy_auc = 0.9099727
    joy_precision = 0.5740741
    joy_recall = 0.57763976
    loss = 0.09940964
    love_accuracy = 0.9789939
    love_auc = 0.96283907
    love_precision = 0.7348485
    love_recall = 0.81512606
    nervousness_accuracy = 0.99576193
    nervousness_auc = 0.7427791
    nervousness_precision = 0.0
    nervousness_recall = 0.0
    neutral_accuracy = 0.7296849
    neutral_auc = 0.8117753
    neutral_precision = 0.56855184
    neutral_recall = 0.74258536
    optimism_accuracy = 0.97236043
    optimism_auc = 0.87280244
    optimism_precision = 0.62857145
    optimism_recall = 0.47311828
    per_example_eval_loss = 0.0995096
    precision_at_threshold_0.00 = 0.041650213
    precision_at_threshold_0.10 = 0.34492928
    precision_at_threshold_0.20 = 0.4777997
    precision_at_threshold_0.30 = 0.5688418
    precision_at_threshold_0.40 = 0.63608086
    precision_at_threshold_0.50 = 0.6950495
    precision_at_threshold_0.60 = 0.7327273
    precision_at_threshold_0.70 = 0.7771429
    precision_at_threshold_0.80 = 0.8079057
    precision_at_threshold_0.90 = 0.8933333
    precision_at_threshold_0.95 = 0.9826087
    precision_at_threshold_0.99 = 0.0
    pride_accuracy = 0.9970518
    pride_auc = 0.689273
    pride_precision = 0.0
    pride_recall = 0.0
    realization_accuracy = 0.97328174
    realization_auc = 0.673867
    realization_precision = 0.0
    realization_recall = 0.0
    recall_at_threshold_0.00 = 1.0
    recall_at_threshold_0.10 = 0.720651
    recall_at_threshold_0.20 = 0.61210304
    recall_at_threshold_0.30 = 0.52614945
    recall_at_threshold_0.40 = 0.45236215
    recall_at_threshold_0.50 = 0.38821298
    recall_at_threshold_0.60 = 0.31837574
    recall_at_threshold_0.70 = 0.25786063
    recall_at_threshold_0.80 = 0.18407331
    recall_at_threshold_0.90 = 0.08468953
    recall_at_threshold_0.95 = 0.035708643
    recall_at_threshold_0.99 = 0.0
    relief_accuracy = 0.9979731
    relief_auc = 0.70399994
    relief_precision = 0.0
    relief_recall = 0.0
    remorse_accuracy = 0.9918924
    remorse_auc = 0.9717813
    remorse_precision = 0.575
    remorse_recall = 0.8214286
    sadness_accuracy = 0.9729132
    sadness_auc = 0.8595727
    sadness_precision = 0.5365854
    sadness_recall = 0.42307693
    sentiment_accuracy = 0.8363079
    sentiment_f1 = 0.68370897
    sentiment_precision = 0.76772606
    sentiment_recall = 0.6162751
    surprise_accuracy = 0.97217613
    surprise_auc = 0.8796486
    surprise_precision = 0.45833334
    surprise_recall = 0.39007092
