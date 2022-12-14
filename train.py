import argparse
import os
import random

import emoji as emotk
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
from transformers import (AdamW, RobertaModel, RobertaTokenizer,
                          get_linear_schedule_with_warmup)

from emoDatasets import GoEmotionDataset
from models import (GoEmotionClassifier, eval_fn, log_metrics, loss_fn,
                    ret_optimizer, ret_scheduler, train_fn)
from train_config import (ekman_mapping, mapping, sentiment_mapping,
                          sweep_config, sweep_defaults, taxon2ekman,
                          taxon2senti)
from utils import inspect_category_wise_data


def parse_args():
    parser = argparse.ArgumentParser(description="loading database information")
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--emoji-rand-init", action="store_true")
    parser.add_argument("--use-emoji", action="store_true")
    parser.add_argument("--sweep-count", type=int, default=1)
    parser.add_argument(
        "--text-align",
        action="store_true",
        help="using alternative text of emojis, which aligns with the textual feature space",
    )
    parser.add_argument("--logdir", type=str, default="exp/")
    parser.add_argument(
        "--model-type", choices=["bert", "roberta", "squeezebert"], default="bert"
    )
    parser.add_argument(
        "--task", choices=["taxon", "ekman", "sentiment"], default="taxon"
    )
    return parser.parse_args()


def tokenize_and_embedize(emo, md, tk):
    emoji_text = emotk.demojize(emo)
    new_str = emoji_text.strip(":").replace("_", " ")
    str_embedding = md.embeddings.word_embeddings(
        torch.Tensor(tk.encode(new_str)).int()
    )[1:-1]
    avg_embedding = torch.mean(str_embedding, dim=0)
    return avg_embedding


def build_dataset(tokenizer_max_len, replace_emoticon):
    train_dataset = GoEmotionDataset(
        train.text.tolist(),
        train[range(n_labels)].values.tolist(),
        tokenizer,
        tokenizer_max_len,
        replace_emoticon=replace_emoticon,
    )
    valid_dataset = GoEmotionDataset(
        valid.text.tolist(),
        valid[range(n_labels)].values.tolist(),
        tokenizer,
        tokenizer_max_len,
        replace_emoticon=replace_emoticon,
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
    model = GoEmotionClassifier(n_train_steps, n_labels, do_prob, bert_model=my_model)
    return model


def one_hot_encoder(df, task_filter):
    one_hot_encoding = []
    for i in tqdm(range(len(df))):
        temp = [0] * n_labels
        label_indices = df.iloc[i]["labels"]
        for index in label_indices:
            # if not index == 27:
            temp[task_filter[index]] = 1
        one_hot_encoding.append(temp)
    return pd.DataFrame(one_hot_encoding)


def trainer(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_dataset, valid_dataset = build_dataset(
            config.tokenizer_max_len, config.replace_emoticon
        )
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
            all_scores = log_metrics(preds, labels, args.task)
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
            with open(sweep_logdir, "a") as f:
                f.write(f"Epoch{epoch}")
                f.write(all_report)
                f.write("\n")
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
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
                print("Model saved as current val_loss is: ", best_val_loss)


if __name__ == "__main__":

    args = parse_args()
    os.makedirs(os.path.join(args.logdir, args.db), exist_ok=True)
    global sweep_logdir
    global exp_dir
    exp_dir = os.path.join(args.logdir, args.db)
    sweep_logdir = os.path.join(args.logdir, args.db, f"{args.db}.log")
    with open(sweep_logdir, "w") as f:
        f.write(
            "See log at \n%s"
            % (f"https://wandb.ai/realzza/{args.db.replace('_', '-')}\n")
        )

    go_emotions = load_dataset("go_emotions")
    data = go_emotions.data
    if args.model_type == "squeezebert":
        model_name = "squeezebert/squeezebert-uncased"
        tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(
            model_name, do_lower_case=True
        )

        my_model = transformers.SqueezeBertModel.from_pretrained(model_name)

    if args.model_type == "bert":
        model_name = "bert-base-cased"
        tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

        my_model = transformers.BertModel.from_pretrained(model_name)

    if args.model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        my_model = RobertaModel.from_pretrained("roberta-base")

    e2v = gensim.models.KeyedVectors.load_word2vec_format(
        "e2vDemo/768-emoji2vec.bin", binary=True
    )

    train, valid, test = (
        data["train"].to_pandas(),
        data["validation"].to_pandas(),
        data["test"].to_pandas(),
    )

    if args.use_emoji:
        all_emojis = set()
        for phase in [train, valid, test]:
            for txt in tqdm(phase["text"]):
                if emotk.emoji_count(txt) > 0:
                    emojis = emotk.emoji_list(txt)
                    for emoji_pair in emojis:
                        all_emojis.add(
                            txt[emoji_pair["match_start"] : emoji_pair["match_end"]]
                        )

        all_emojis = list(all_emojis)

        if args.emoji_rand_init:
            num_added_tokens = tokenizer.add_tokens(all_emojis)
            print("%d emojis added" % num_added_tokens)
            my_model.resize_token_embeddings(
                len(tokenizer)
            )  # https://huggingface.co/docs/transformers/internal/tokenization_utils?highlight=add_token#transformers.SpecialTokensMixin.add_tokens
        else:
            if args.text_align:
                for emoji in all_emojis:
                    emoji_embd = tokenize_and_embedize(emoji, my_model, tokenizer)
                    tokenizer.add_tokens(emoji)
                    my_model.resize_token_embeddings(len(tokenizer))
                    with torch.no_grad():
                        my_model.embeddings.word_embeddings.weight[-1, :] = emoji_embd

            else:
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
                        my_model.resize_token_embeddings(len(tokenizer))
                        with torch.no_grad():
                            my_model.embeddings.word_embeddings.weight[
                                -1, :
                            ] = emoji_embd
                    else:

                        tokenizer.add_tokens(emoji)
                        my_model.resize_token_embeddings(len(tokenizer))
                        print(i, emoji, len(tokenizer))

    print(train.shape, valid.shape, test.shape)

    import wandb

    wandb.login()
    if args.task == "ekman" or args.task == "sentiment":
        sweep_config["parameters"]["learning_rate"]["values"] = [4e-6, 2e-6]
    sweep_id = wandb.sweep(sweep_config, project=args.db)

    # import pdb; pdb.set_trace()

    if args.task == "taxon":
        task_filter = {i: i for i in range(28)}
    elif args.task == "ekman":
        task_filter = taxon2ekman
        mapping = ekman_mapping
    elif args.task == "sentiment":
        task_filter = taxon2senti
        mapping = sentiment_mapping

    n_labels = len(mapping)
    train_ohe_labels = one_hot_encoder(train, task_filter)
    valid_ohe_labels = one_hot_encoder(valid, task_filter)
    test_ohe_labels = one_hot_encoder(test, task_filter)

    train = pd.concat([train, train_ohe_labels], axis=1)
    valid = pd.concat([valid, valid_ohe_labels], axis=1)
    test = pd.concat([test, test_ohe_labels], axis=1)

    sample_train_dataset, _ = build_dataset(40, False)
    len(sample_train_dataset)

    print(sweep_id)
    wandb.agent(sweep_id, function=trainer, count=args.sweep_count)
