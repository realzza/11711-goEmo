import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics, model_selection, preprocessing
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import wandb

taxon_mapping = {
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


def ret_optimizer(model):
    """
    Taken from Abhishek Thakur's Tez library example:
    https://github.com/abhishekkrthakur/tez/blob/main/examples/text_classification/binary.py
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = AdamW(optimizer_parameters, lr=wandb.config.learning_rate)
    return opt


def ret_scheduler(optimizer, num_train_steps):
    sch = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    return sch


def loss_fn(outputs, labels):
    if labels is None:
        return None
    return nn.BCEWithLogitsLoss()(outputs, labels.float())


def log_metrics(preds, labels, task="taxon"):
    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()

    """
    auc_micro_list = []
    for i in range(n_labels):
      current_pred = preds.T[i]
      current_label = labels.T[i]
      fpr_micro, tpr_micro, _ = metrics.roc_curve(current_label.T, current_pred.T)
      auc_micro = metrics.auc(fpr_micro, tpr_micro)
      auc_micro_list.append(auc_micro)
    
    return {"auc": np.array(auc_micro).mean()}
    """

    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())

    auc_micro = metrics.auc(fpr_micro, tpr_micro)
    discrete_preds = np.where(preds > 0.3, 1, 0)
    if task == "taxon":
        mapping = taxon_mapping
    elif task == "ekman":
        mapping = ekman_mapping
    elif task == "sentiment":
        mapping = sentiment_mapping
    label_names = list(mapping.values())
    all_report = metrics.classification_report(
        labels.astype(int), discrete_preds, target_names=label_names
    )
    return {"auc_micro": auc_micro, "all_report": all_report}
