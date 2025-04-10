import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X, y, categorical_cols):
        self.categorical = torch.LongTensor(X.loc[:, categorical_cols].values)
        self.numerical = torch.FloatTensor(X.drop(columns=categorical_cols).values)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.numerical[idx], self.categorical[idx], self.y[idx]

class TABMLP(nn.Module):
    def __init__(self, numeric_count, emb_dims, num_classes=2):
        super(TABMLP, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, emb_dim) for cat_size, emb_dim in emb_dims])
        emb_dim_total = sum([emb_dim for _, emb_dim in emb_dims])

        self.fc1 = nn.Linear(emb_dim_total + numeric_count, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 40)
        self.fc4 = nn.Linear(40, num_classes)

        self.bn1 = nn.BatchNorm1d(emb_dim_total + numeric_count)
        self.bn2 = nn.BatchNorm1d(160)
        self.bn3 = nn.BatchNorm1d(80)
        self.bn4 = nn.BatchNorm1d(40)

        self.drop = nn.Dropout(0.2)

    def forward(self, x_num, x_cat):
        if self.embeddings:
            x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_emb = torch.cat(x_emb, 1)
            x = torch.cat([x_emb, x_num], 1)
        else:
            x = x_num

        x = self.bn1(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.drop(x)

        x = F.relu(self.bn3(self.fc2(x)))
        x = self.drop(x)

        x = F.relu(self.bn4(self.fc3(x)))
        x = self.drop(x)

        x = self.fc4(x)
        return x

def plot_loss(train_losses, val_losses):
    clear_output(wait=True)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

@torch.no_grad()
def test(model, loader, criterion, device):
    loss_log = []
    acc_log = []
    model.eval()
    for x_num, x_cat, target in tqdm(loader, desc="Validating"):
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        target = target.to(device)

        logits = model(x_num, x_cat)
        loss = criterion(logits, target)
        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == target).sum() / len(x_num)
        acc_log.append(acc.item())
    return np.mean(loss_log)

def train_epoch(model, optimizer, loader, criterion, device):
    loss_log = []
    acc_log = []
    model.train()
    for x_num, x_cat, target in tqdm(loader, desc="Training"):
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())

        acc = (logits.argmax(dim=1) == target).sum() / len(x_num)
        acc_log.append(acc.item())

    return np.mean(loss_log)

def train(model, optimizer, n_epochs, train_loader, val_loader, criterion, device, scheduler=None):
    train_loss_log = []
    val_loss_log = []
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, optimizer, train_loader, criterion, device)
        val_loss = test(model, val_loader, criterion, device)
        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        plot_loss(train_loss_log, val_loss_log)
        if scheduler is not None:
            scheduler.step(val_loss)
        print(f"Epoch {epoch}: train loss = {train_loss:.4f}; valid loss = {val_loss:.4f}")


@torch.no_grad()
def evaluate(model, loader, device, multi=False):
    model.eval()
    all_targets = []
    all_probs = []
    all_preds = []
    for x_num, x_cat, target in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        target = target.to(device)
        logits = model(x_num, x_cat)
        if multi:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_targets.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    return np.array(all_targets), np.array(all_probs), np.array(all_preds)