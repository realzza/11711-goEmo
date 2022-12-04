import os
import random

import emoji
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from sklearn import metrics, model_selection, preprocessing
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from emoDatasets import GoEmotionDataset
from models import (GoEmotionClassifier, eval_fn, log_metrics, loss_fn,
                    ret_optimizer, ret_scheduler, train_fn)
# wandb.login()
from train_config import mapping, sweep_config, sweep_defaults
from utils import inspect_category_wise_data

# import wandb


def build_dataset(tokenizer_max_len):
    train_dataset = GoEmotionDataset(
        train.text.tolist(),
        train[range(n_labels)].values.tolist(),
        tokenizer,
        tokenizer_max_len,
    )
    valid_dataset = GoEmotionDataset(
        valid.text.tolist(),
        valid[range(n_labels)].values.tolist(),
        tokenizer,
        tokenizer_max_len,
    )

    return train_dataset, valid_dataset


def build_dataloader(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    return train_data_loader, valid_data_loader


def ret_model(n_train_steps, do_prob):
    model = GoEmotionClassifier(n_train_steps, n_labels, do_prob, bert_model=bert_model)
    return model


def one_hot_encoder(df):
    one_hot_encoding = []
    for i in tqdm(range(len(df))):
        temp = [0] * n_labels
        label_indices = df.iloc[i]["labels"]
        for index in label_indices:
            temp[index] = 1
        one_hot_encoding.append(temp)
    return pd.DataFrame(one_hot_encoding)


def trainer(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_dataset, valid_dataset = build_dataset(config.tokenizer_max_len)
        train_data_loader, valid_data_loader = build_dataloader(
            train_dataset, valid_dataset, config.batch_size
        )
        print("Length of Train Dataloader: ", len(train_data_loader))
        print("Length of Valid Dataloader: ", len(valid_data_loader))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_train_steps = int(len(train_dataset) / config.batch_size * 10)

        model = ret_model(n_train_steps, config.dropout)
        optimizer = ret_optimizer(model)
        scheduler = ret_scheduler(optimizer, n_train_steps)
        model.to(device)
        model = nn.DataParallel(model)
        wandb.watch(model)

        n_epochs = config.epochs

        best_val_loss = 100
        for epoch in tqdm(range(n_epochs)):
            train_loss = train_fn(
                train_data_loader, model, optimizer, device, scheduler
            )
            eval_loss, preds, labels = eval_fn(valid_data_loader, model, device)

            # import pdb; pdb.set_trace()
            all_scores = log_metrics(preds, labels)
            auc_score = all_scores["auc_micro"]
            all_report = all_scores["all_report"]
            macro_avg_precision, macro_avg_recall, macro_avg_f1 = map(
                float, all_report.split("\n")[-4].split()[2:5]
            )
            print(
                "AUC score: %.4f\nPrecision macro: %.4f\nRecall macro: %.4f\nF1 macro: %.4f"
                % (auc_score, macro_avg_precision, macro_avg_recall, macro_avg_f1)
            )
            print(all_report)
            avg_train_loss, avg_val_loss = train_loss / len(
                train_data_loader
            ), eval_loss / len(valid_data_loader)
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "auc_score": auc_score,
                    "macro_precision": macro_avg_precision,
                    "macro_recall": macro_avg_recall,
                    "macro_f1": macro_avg_f1,
                    "batch_size": config.batch_size,
                    "dropout": config.dropout,
                    "lr": config.learning_rate,
                }
            )
            print("Average Train loss: ", avg_train_loss)
            print("Average Valid loss: ", avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "./best_model.pt")
                print("Model saved as current val_loss is: ", best_val_loss)


if __name__ == "__main__":

    go_emotions = load_dataset("go_emotions")
    data = go_emotions.data
    tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(
        "squeezebert/squeezebert-uncased", do_lower_case=True
    )

    bert_model = transformers.SqueezeBertModel.from_pretrained(
        "squeezebert/squeezebert-uncased"
    )

    e2v = gensim.models.KeyedVectors.load_word2vec_format(
        "e2vDemo/768-emoji2vec.bin", binary=True
    )

    train, valid, test = (
        data["train"].to_pandas(),
        data["validation"].to_pandas(),
        data["test"].to_pandas(),
    )

    all_emojis = set()
    for phase in [train, valid, test]:
        for txt in tqdm(phase["text"]):
            if emoji.emoji_count(txt) > 0:
                # print(txt)
                emojis = emoji.emoji_list(txt)
                for emoji_pair in emojis:
                    all_emojis.add(
                        txt[emoji_pair["match_start"] : emoji_pair["match_end"]]
                    )

    all_emojis = list(all_emojis)
    error_emojis = []
    health_emojis = []
    for emoji in all_emojis:
        try:
            tmp_emoji = e2v[emoji[0]]
            health_emojis.append(emoji[0])
        except:
            error_emojis.append(emoji[0])

    for i, emoji in enumerate(all_emojis):
        emoji = emoji[0]
        if emoji in health_emojis:
            emoji_embd = torch.Tensor(e2v[emoji])
            tokenizer.add_tokens(emoji)
            bert_model.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                bert_model.embeddings.word_embeddings.weight[-1, :] = emoji_embd
        else:

            # import pdb; pdb.set_trace()
            tokenizer.add_tokens(emoji)
            bert_model.resize_token_embeddings(len(tokenizer))
            print(i, emoji, len(tokenizer))

    print(train.shape, valid.shape, test.shape)

    import wandb

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="cpu-test")
    n_labels = len(mapping)

    train_ohe_labels = one_hot_encoder(train)
    valid_ohe_labels = one_hot_encoder(valid)
    test_ohe_labels = one_hot_encoder(test)

    train = pd.concat([train, train_ohe_labels], axis=1)
    valid = pd.concat([valid, valid_ohe_labels], axis=1)
    test = pd.concat([test, test_ohe_labels], axis=1)

    sample_train_dataset, _ = build_dataset(40)
    # print(sample_train_dataset[0])
    len(sample_train_dataset)

    print(sweep_id)
    wandb.agent(sweep_id, function=trainer, count=1)
