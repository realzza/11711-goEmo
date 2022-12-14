# External dependencies
import os
import pickle as pk

import wandb

# Internal dependencies
from parameter_parser import CliParser
from model import Emoji2Vec
import torch
from utils import (
    build_kb,
    get_examples_from_kb,
    generate_embeddings,
    get_metrics,
    generate_predictions,
)


# Execute training sequence
def __run_training():
    # Setup
    args = CliParser()
    args.print_params("EMOJI TRAINING")

    # Build knowledge base
    print("reading training data from: " + args.data_folder)
    kb, ind2phr, ind2emoji = build_kb(args.data_folder)

    # Save the mapping from index to emoji
    pk.dump(ind2emoji, open(args.mapping_file, "wb"))

    # Get the embeddings for each phrase in the training set
    embeddings_array = generate_embeddings(
        ind2phr=ind2phr,
        kb=kb,
        embeddings_file=args.embeddings_file,
        word2vec_file=args.word2vec_file,
    )

    # Get examples of each example type in two sets. This is just a reprocessing of the knowledge base for efficiency,
    # so we don't have to generate the train and dev set on each train
    train_set = get_examples_from_kb(kb=kb, example_type="train")
    dev_set = get_examples_from_kb(kb=kb, example_type="dev")

    train_save_evaluate(
        params=args.model_params,
        kb=kb,
        train_set=train_set,
        dev_set=dev_set,
        ind2emoji=ind2emoji,
        embeddings_array=embeddings_array,
        dataset_name=args.dataset,
    )


def train_save_evaluate(
        params, kb, train_set, dev_set, ind2emoji, embeddings_array, dataset_name
):
    # If the minibatch is larger than the number of emojis we have, we can't fill train/test batches
    if params.mb > len(ind2emoji):
        print(
            str.format(
                "Skipping: k={}, batch={}, epochs={}, ratio={}, dropout={}",
                params.out_dim,
                params.pos_ex,
                params.max_epochs,
                params.neg_ratio,
                params.dropout,
            )
        )
        print("Can't have an mb > len(ind2emoji)")
        return "N/A"
    else:
        print(
            str.format(
                "Training: k={}, batch={}, epochs={}, ratio={}, dropout={}",
                params.out_dim,
                params.pos_ex,
                params.max_epochs,
                params.neg_ratio,
                params.dropout,
            )
        )

    model_folder = params.model_folder(dataset_name=dataset_name)
    model_path = model_folder + "/model.pt"

    dsets = {"train": train_set, "dev": dev_set}
    predictions = dict()
    results = dict()

    from wandb_utils import init_wandb
    os.makedirs(model_folder, exist_ok=True)
    init_wandb(model_folder)

    def end_of_epoch_validation_callback():
        torch.save(model.nn, model_folder + "/model.pt")

        if params.in_dim != params.out_dim:
            embd_array = model.nn.project_embeddings(embeddings_array)

        e2v = model.create_gensim_files(
            model_folder=model_folder,
            ind2emoj=ind2emoji,
            out_dim=params.out_dim,
        )

        data = dsets['dev']
        _, pred_values, _, true_values = generate_predictions(
            e2v=e2v,
            dset=data,
            phr_embeddings=embd_array,
            ind2emoji=ind2emoji,
            threshold=params.class_threshold,
        )

        pk.dump(predictions, open(model_folder + "/results.p", "wb"))

        true_labels = [bool(x) for x in true_values]
        pred_labels = [
            x >= params.class_threshold for x in pred_values
        ]
        # Calculate metrics
        acc, f1, auc = get_metrics(
            pred_labels, pred_values, true_labels, true_values
        )
        print(
            f"Validation Accuracy(>{params.class_threshold}): {acc}, f1: {f1}, auc: {auc}"
        )
        wandb.log(
            dict(val_acc=acc, val_f1=f1, val_auc=auc)
        )

        results['dev'] = {"accuracy": acc, "f1": f1, "auc": auc}

    model = Emoji2Vec(
        model_params=params,
        num_emojis=kb.dim_size(0),
        embeddings_array=embeddings_array,
    )
    model.train(
        kb=kb, epochs=params.max_epochs, learning_rate=params.learning_rate,
        end_of_epoch_callback=end_of_epoch_validation_callback,
    )

    return results["dev"]


if __name__ == "__main__":
    __run_training()
