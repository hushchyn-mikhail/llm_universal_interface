import math

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels = torch.stack(labels)
    return data_padded, labels


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            input_dim=1,
            hidden_dim=64,
            num_layers=1,
            num_classes=2
        ):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

class TransformerClassifier(nn.Module):
    def __init__(
            self, 
            input_dim=1, 
            embed_dim=256, 
            num_heads=8, 
            num_layers=3, 
            num_classes=2, 
            dropout=0, 
            maxlen=20
        ):
        super(TransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(emb_size=embed_dim, dropout=dropout, maxlen=maxlen)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x.mean(dim=1)
        return self.fc(out)


def plot_loss(train_losses, val_losses):
    clear_output(wait=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

@torch.no_grad()
def test(model, loader, criterion, device):
    loss_log = []
    model.eval()
    for data, target in tqdm(loader, desc="Validating"):
        data = data.to(device)
        target = target.to(device)
        
        logits = model(data)
        loss = criterion(logits, target)
        loss_log.append(loss.item())
        
    return np.mean(loss_log)

def train_epoch(model, optimizer, train_loader, criterion, device):
    loss_log = []
    model.train()
    for data, target in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        loss_log.append(loss.item())
  
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
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        logits = model(data)
        if multi:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_targets.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    return np.array(all_targets), np.array(all_probs), np.array(all_preds)