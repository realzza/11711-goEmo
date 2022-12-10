import argparse
import os
import numpy as np
import emoji
import gensim
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from emoDatasets import GoEmotionDataset
from models import GoEmotionClassifier, log_metrics
from train_config import (ekman_mapping, mapping, sentiment_mapping, taxon2ekman, taxon2senti)
from sklearn import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="loading database information")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--emoji-rand-init", action="store_true")
    parser.add_argument("--use-emoji", action="store_true")
    parser.add_argument("--logdir", type=str, default="exp/")
    parser.add_argument(
        "--model-type", choices=["bert", "roberta", "squeezebert"], default="bert"
    )
    parser.add_argument(
        "--task", choices=["taxon", "ekman", "sentiment"], default="taxon"
    )
    return parser.parse_args()


def one_hot_encoder(df, task_filter):
    one_hot_encoding = []
    for i in tqdm(range(len(df))):
        temp = [0] * n_labels
        label_indices = df.iloc[i]["labels"]
        for index in label_indices:
            if not index == 27:
                temp[task_filter[index]] = 1
        one_hot_encoding.append(temp)
    return pd.DataFrame(one_hot_encoding)


def inference(model, device, dataloader):
    token_ids = []
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            token_ids.extend(ids)
            labels.extend(targets)
            preds.extend(torch.sigmoid(outputs))

    return token_ids, preds, labels


def inference_and_test(config):
    test_dataset = GoEmotionDataset(
        test.text.tolist(),
        test[range(n_labels)].values.tolist(),
        tokenizer,
        config['tokenizer_max_len'],
        replace_emoticon=False,
    )
    dataloader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1
    )
    print("Length of Test Dataloader: ", len(dataloader))

    sentences, preds, labels = inference(model, device, dataloader)
    sentences = test_dataset.tokenizer.batch_decode(sentences)
    sentences = [
        s.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').replace('<s>', '').replace('</s>', '')
        .replace('<pad>', '') for s in sentences
    ]

    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()
    discrete_preds = np.where(preds > 0.3, 1, 0)

    # log metrics
    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    auc_micro = metrics.auc(fpr_micro, tpr_micro)
    label_names = list(mapping.values())
    all_report = metrics.classification_report(
        labels.astype(int), discrete_preds, target_names=label_names
    )

    macro_avg_precision, macro_avg_recall, macro_avg_f1 = map(
        float, all_report.split("\n")[-4].split()[2:5]
    )
    print(
        f"AUC score: {auc_micro:.4f}\nPrecision macro: {macro_avg_precision:.4f}\n"
        f"Recall macro: {macro_avg_recall:.4f}\nF1 macro: {macro_avg_f1:.4f}"
    )
    print(all_report)
    with open(log_file, "a") as f:
        f.write(f'{all_report}\n')

    # save predictions
    assert len(sentences) == len(discrete_preds) == len(labels)
    with open(pred_file, "w") as f:
        for i, s in enumerate(sentences):
            pred = ','.join(
                [mapping[ei] for ei, p in enumerate(discrete_preds[i]) if p == 1]
            )
            label = ','.join(
                [mapping[ei] for ei, p in enumerate(labels[i]) if p == 1]
            )
            f.write(f'{s}\t{pred}\t{label}\n')


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.join(args.logdir, args.db), exist_ok=True)
    global log_file
    global exp_dir
    exp_dir = os.path.join(args.logdir, args.db)
    log_file = os.path.join(args.logdir, args.db, f"{args.db}.log")
    pred_file = os.path.join(args.logdir, args.db, "pred.txt")

    go_emotions = load_dataset("go_emotions")
    data = go_emotions.data

    if args.model_type == "squeezebert":
        model_name = "squeezebert/squeezebert-uncased"
        tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(model_name, do_lower_case=True)
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
                if emoji.emoji_count(txt) > 0:
                    emojis = emoji.emoji_list(txt)
                    for emoji_pair in emojis:
                        all_emojis.add(
                            txt[emoji_pair["match_start"]: emoji_pair["match_end"]]
                        )

        all_emojis = list(all_emojis)

        if args.emoji_rand_init:
            num_added_tokens = tokenizer.add_tokens(all_emojis)
            print("%d emojis added" % num_added_tokens)
            my_model.resize_token_embeddings(
                len(tokenizer)
            )  # https://huggingface.co/docs/transformers/internal/tokenization_utils?highlight=add_token#transformers.SpecialTokensMixin.add_tokens
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
                        my_model.embeddings.word_embeddings.weight[-1, :] = emoji_embd
                else:

                    tokenizer.add_tokens(emoji)
                    my_model.resize_token_embeddings(len(tokenizer))
                    print(i, emoji, len(tokenizer))

    print(test.shape)

    if args.task == "taxon":
        task_filter = {i: i for i in range(28)}
    elif args.task == "ekman":
        task_filter = taxon2ekman
        mapping = ekman_mapping
    elif args.task == "sentiment":
        task_filter = taxon2senti
        mapping = sentiment_mapping

    n_labels = len(mapping)
    if args.split == 'val':
        test = valid

    test_ohe_labels = one_hot_encoder(test, task_filter)

    test = pd.concat([test, test_ohe_labels], axis=1)

    # load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoEmotionClassifier(0, n_labels, 0, bert_model=my_model)
    model.to(device)
    model = nn.DataParallel(model)
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

    inference_and_test(
        dict(
            batch_size=64,
            tokenizer_max_len=40,
        )
    )
