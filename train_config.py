sweep_config = {
    "method": "random",  # grid, random, bayesian
    "metric": {"name": "macro_recall", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            # 'values': [3e-5]
            # default value for taxon, will adapt ekman and sentiment in train.py
            "values": [5e-5, 3e-5]
        },
        "batch_size": {
            # 'values': [32]
            "values": [32, 64]
        },
        "epochs": {"value": 10},
        "dropout": {
            # 'values': [0.5]
            "values": [0.3, 0.5]
            # "values": [0.3, 0.4, 0.5]
        },
        "tokenizer_max_len": {"value": 40},
        "replace_emoticon": {"value": False},
    },
}

sweep_defaults = {
    "learning_rate": 3e-5,
    "batch_size": 64,
    "epochs": 10,
    "dropout": 0.3,
    "tokenizer_max_len": 40,
}


mapping = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

ekman_ = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}

senti_ = {
    "positive": [
        "amusement",
        "excitement",
        "joy",
        "love",
        "desire",
        "optimism",
        "caring",
        "pride",
        "admiration",
        "gratitude",
        "relief",
        "approval",
    ],
    "negative": [
        "fear",
        "nervousness",
        "remorse",
        "embarrassment",
        "disappointment",
        "sadness",
        "grief",
        "disgust",
        "anger",
        "annoyance",
        "disapproval",
    ],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
}

taxon2ekman = {}

for emoId in mapping:
    ekid = 0
    for ekemo in ekman_:
        if mapping[emoId] in ekman_[ekemo]:
            taxon2ekman[emoId] = ekid
        ekid += 1

taxon2senti = {}

for emoId in mapping:
    sentid = 0
    for sentiemo in senti_:
        if mapping[emoId] in senti_[sentiemo]:
            taxon2senti[emoId] = sentid
        sentid += 1

ekman_mapping = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "sadness",
    5: "surprise",
}

sentiment_mapping = {
    0: "positive",
    1: "negative",
    2: "ambiguous",
}
