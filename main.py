#!/usr/bin/env python3

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from dataset import BldgDataset
from model import MLP


def main():
    model_name = 'test_model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    EPOCHS = 50
    lr = 5e-3
    l2_reg = 0.1

    data = pd.read_csv('data/data_df.csv', header=0)

    # TODO: Replace with cross validation
    test_df = data.sample(frac=0.2)
    data.drop(test_df.index, inplace=True)

    train_data = BldgDataset(data)
    test_data = BldgDataset(test_df)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size*2,
                             shuffle=False)

    model = MLP(in_feats=train_data.num_feats(),
                hidden_dim=10,
                n_classes=train_data.num_classes())
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=l2_reg)

    model.to(device)
    for epoch in range(EPOCHS):

        data_iter = tqdm(
            train_loader,
            desc=f'Epoch: {epoch:02}',
            total=len(train_loader)
        )

        model.train()
        avg_loss = 0.
        for i, (X, y) in enumerate(data_iter):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(X)
            # TODO: Find a loss function for ordinal classification
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            data_iter.set_postfix({
                'avg_loss': avg_loss / (i+1)
            })

    y_pred, y_true = [], []
    with torch.no_grad():
        model.eval()
        for X, y in test_loader:
            X = X.to(device)

            preds = torch.argmax(model(X), dim=1)
            y_pred.extend(preds.cpu())
            y_true.extend(y)

    # TODO: Paper used MSE not accuracy
    print(f'Test Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    torch.save(model.state_dict(), f'models/{model_name}.pt')


if __name__ == '__main__':
    main()
