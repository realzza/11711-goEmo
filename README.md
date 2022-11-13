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

## Baseline Results

```bash
cd goemotions
python calculate_metrics.py --test_data data/test.tsv --predictions ../exp/test.tsv.predictions.tsv
```

Threshold = 0.3

```json
{
  "accuracy": 0.41533075363921135,
  "macro_precision": 0.416860369666073,
  "macro_recall": 0.35040055141550974,
  "macro_f1": 0.3631723079864065,
  "micro_precision": 0.5688418175606423,
  "micro_recall": 0.5261494706904725,
  "micro_f1": 0.5466633834031027,
  "weighted_precision": 0.526415219464872,
  "weighted_recall": 0.5261494706904725,
  "weighted_f1": 0.509780376274107,
  "admiration_accuracy": 0.9303482587064676,
  "admiration_precision": 0.6334745762711864,
  "admiration_recall": 0.5932539682539683,
  "admiration_f1": 0.6127049180327868,
  "amusement_accuracy": 0.9751243781094527,
  "amusement_precision": 0.7034700315457413,
  "amusement_recall": 0.8446969696969697,
  "amusement_f1": 0.7676419965576592,
  "anger_accuracy": 0.9585406301824212,
  "anger_precision": 0.4262295081967213,
  "anger_recall": 0.3939393939393939,
  "anger_f1": 0.40944881889763785,
  "annoyance_accuracy": 0.9290584116454763,
  "annoyance_precision": 0.328042328042328,
  "annoyance_recall": 0.19375,
  "annoyance_f1": 0.24361493123772102,
  "approval_accuracy": 0.9235304956697992,
  "approval_precision": 0.35046728971962615,
  "approval_recall": 0.21367521367521367,
  "approval_f1": 0.2654867256637168,
  "caring_accuracy": 0.9695964621337756,
  "caring_precision": 0.2972972972972973,
  "caring_recall": 0.16296296296296298,
  "caring_f1": 0.2105263157894737,
  "confusion_accuracy": 0.968859406670352,
  "confusion_precision": 0.40476190476190477,
  "confusion_recall": 0.2222222222222222,
  "confusion_f1": 0.2869198312236287,
  "curiosity_accuracy": 0.9449051041090842,
  "curiosity_precision": 0.48021108179419525,
  "curiosity_recall": 0.6408450704225352,
  "curiosity_f1": 0.5490196078431372,
  "desire_accuracy": 0.9848903629998157,
  "desire_precision": 0.5142857142857142,
  "desire_recall": 0.21686746987951808,
  "desire_f1": 0.3050847457627119,
  "disappointment_accuracy": 0.9721761562557583,
  "disappointment_precision": 0.5,
  "disappointment_recall": 0.006622516556291391,
  "disappointment_f1": 0.0130718954248366,
  "disapproval_accuracy": 0.9384558688041275,
  "disapproval_precision": 0.2919254658385093,
  "disapproval_recall": 0.1760299625468165,
  "disapproval_f1": 0.2196261682242991,
  "disgust_accuracy": 0.978441127694859,
  "disgust_precision": 0.5405405405405406,
  "disgust_recall": 0.3252032520325203,
  "disgust_f1": 0.40609137055837563,
  "embarrassment_accuracy": 0.9931822369633315,
  "embarrassment_precision": 0.0,
  "embarrassment_recall": 0.0,
  "embarrassment_f1": 0.0,
  "excitement_accuracy": 0.9810208218168417,
  "excitement_precision": 0.5,
  "excitement_recall": 0.2815533980582524,
  "excitement_f1": 0.36024844720496896,
  "fear_accuracy": 0.9887599041827898,
  "fear_precision": 0.6976744186046512,
  "fear_recall": 0.38461538461538464,
  "fear_f1": 0.49586776859504134,
  "gratitude_accuracy": 0.9896812235120693,
  "gratitude_precision": 0.9277456647398844,
  "gratitude_recall": 0.9119318181818182,
  "gratitude_f1": 0.9197707736389685,
  "grief_accuracy": 0.9988944168048646,
  "grief_precision": 0.0,
  "grief_recall": 0.0,
  "grief_f1": 0.0,
  "joy_accuracy": 0.9747558503777409,
  "joy_precision": 0.5740740740740741,
  "joy_recall": 0.577639751552795,
  "joy_f1": 0.5758513931888545,
  "love_accuracy": 0.9789939192924267,
  "love_precision": 0.7348484848484849,
  "love_recall": 0.8151260504201681,
  "love_f1": 0.7729083665338646,
  "nervousness_accuracy": 0.9957619310853142,
  "nervousness_precision": 0.0,
  "nervousness_recall": 0.0,
  "nervousness_f1": 0.0,
  "optimism_accuracy": 0.9723604201216142,
  "optimism_precision": 0.6285714285714286,
  "optimism_recall": 0.4731182795698925,
  "optimism_f1": 0.5398773006134969,
  "pride_accuracy": 0.9970517781463055,
  "pride_precision": 0.0,
  "pride_recall": 0.0,
  "pride_f1": 0.0,
  "realization_accuracy": 0.9732817394508937,
  "realization_precision": 0.0,
  "realization_recall": 0.0,
  "realization_f1": 0.0,
  "relief_accuracy": 0.997973097475585,
  "relief_precision": 0.0,
  "relief_recall": 0.0,
  "relief_f1": 0.0,
  "remorse_accuracy": 0.9918923899023402,
  "remorse_precision": 0.575,
  "remorse_recall": 0.8214285714285714,
  "remorse_f1": 0.676470588235294,
  "sadness_accuracy": 0.9729132117191819,
  "sadness_precision": 0.5365853658536586,
  "sadness_recall": 0.4230769230769231,
  "sadness_f1": 0.4731182795698925,
  "surprise_accuracy": 0.9721761562557583,
  "surprise_precision": 0.4583333333333333,
  "surprise_recall": 0.3900709219858156,
  "surprise_f1": 0.421455938697318,
  "neutral_accuracy": 0.7296849087893864,
  "neutral_precision": 0.5685518423307626,
  "neutral_recall": 0.7425853385562395,
  "neutral_f1": 0.6440184421256976
}
```

## Baseline results ordered by f1 score

```json
{
  "embarrassment_f1": 0.0, 
  "grief_f1": 0.0, 
  "nervousness_f1": 0.0, 
  "pride_f1": 0.0, 
  "realization_f1": 0.0, 
  "relief_f1": 0.0, 
  "disappointment_f1": 0.0130718954248366, 
  "caring_f1": 0.2105263157894737, 
  "disapproval_f1": 0.2196261682242991, 
  "annoyance_f1": 0.24361493123772102, 
  "approval_f1": 0.2654867256637168, 
  "confusion_f1": 0.2869198312236287, 
  "desire_f1": 0.3050847457627119, 
  "excitement_f1": 0.36024844720496896, 
  "macro_f1": 0.3631723079864065, 
  "disgust_f1": 0.40609137055837563, 
  "anger_f1": 0.40944881889763785, 
  "surprise_f1": 0.421455938697318, 
  "sadness_f1": 0.4731182795698925, 
  "fear_f1": 0.49586776859504134, 
  "weighted_f1": 0.509780376274107, 
  "optimism_f1": 0.5398773006134969, 
  "micro_f1": 0.5466633834031027, 
  "curiosity_f1": 0.5490196078431372, 
  "joy_f1": 0.5758513931888545, 
  "admiration_f1": 0.6127049180327868, 
  "neutral_f1": 0.6440184421256976,
  "remorse_f1": 0.676470588235294, 
  "amusement_f1": 0.7676419965576592, 
  "love_f1": 0.7729083665338646, 
  "gratitude_f1": 0.9197707736389685
}
```