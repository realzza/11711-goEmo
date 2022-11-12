import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def loss_fn(outputs, labels):
    if labels is None:
        return None
    return nn.BCEWithLogitsLoss()(outputs, labels.float())


def train_fn(data_loader, model, optimizer, device, scheduler):
    '''
        Modified from Abhishek Thakur's BERT example: 
        https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py
    '''

    train_loss = 0.0
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)

        loss = loss_fn(outputs, targets)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
    return train_loss
    

def eval_fn(data_loader, model, device):
    '''
        Modified from Abhishek Thakur's BERT example: 
        https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py
    '''
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs, targets)
            eval_loss += loss.item()
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))
    return eval_loss, fin_outputs, fin_targets


# def trainer(config=None):
#     with wandb.init(config=config):
#         config = wandb.config

#         train_dataset, valid_dataset = build_dataset(config.tokenizer_max_len)
#         train_data_loader, valid_data_loader = build_dataloader(train_dataset, valid_dataset, config.batch_size)
#         print("Length of Train Dataloader: ", len(train_data_loader))
#         print("Length of Valid Dataloader: ", len(valid_data_loader))

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         n_train_steps = int(len(train_dataset) / config.batch_size * 10)

#         model = ret_model(n_train_steps, config.dropout)
#         optimizer = ret_optimizer(model)
#         scheduler = ret_scheduler(optimizer, n_train_steps)
#         model.to(device)
#         model = nn.DataParallel(model)
#         wandb.watch(model)
        
#         n_epochs = config.epochs

#         best_val_loss = 100
#         for epoch in tqdm(range(n_epochs)):
#             train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
#             eval_loss, preds, labels = eval_fn(valid_data_loader, model, device)
          
#             auc_score = log_metrics(preds, labels)["auc_micro"]
#             print("AUC score: ", auc_score)
#             avg_train_loss, avg_val_loss = train_loss / len(train_data_loader), eval_loss / len(valid_data_loader)
#             wandb.log({
#                 "epoch": epoch + 1,
#                 "train_loss": avg_train_loss,
#                 "val_loss": avg_val_loss,
#                 "auc_score": auc_score,
#             })
#             print("Average Train loss: ", avg_train_loss)
#             print("Average Valid loss: ", avg_val_loss)

#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 torch.save(model.state_dict(), "./best_model.pt")  
#                 print("Model saved as current val_loss is: ", best_val_loss)    