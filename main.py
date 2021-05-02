#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from dataset import BldgDataset
from model import ClassificationModel, RegressionModel


def main():
    model_name = 'model_test_m8_cols'
    model_type = 'classification'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    EPOCHS = 25
    lr = 3e-3
    l2_reg = 0.1

    data = pd.read_csv('data/preprocessed.csv', header=0)
    model_8_cols = [
        'hazard',
        'secondary_hazard_31',
        # 'secondary_hazard_32',
        'event_size_1',
        'event_size_2',
        'event_size_3',
        'median_year_built',
        'occupancy',
        'walls',
        'roofing',
        'roof_type',
        'height_1',
        'height_2',
        'height_3',
        'height_6',
        'percent_owner_occupied',
        'percent_renter_occupied',
        'roughness_2',
        # 'roughness_3',
        'pop',
        'forested',
        'housing_density'
    ]

    data = data[model_8_cols + ['target']]

    # TODO: Replace with cross validation?
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

    if model_type == 'regression':
        model = RegressionModel(in_feats=train_data.num_feats(),
                                hidden_dim=10,
                                dropout=0.2,
                                activation='relu')
    else:
        model = ClassificationModel(in_feats=train_data.num_feats(),
                                    hidden_dim=10,
                                    n_classes=train_data.num_classes(),
                                    dropout=0.2,
                                    activation='relu')
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
            if model_type == 'regression':
                loss = F.mse_loss(logits.squeeze(), y.type(torch.float32))
            else:
                loss = F.cross_entropy(logits, y)
                # y_one_hot = F.one_hot(y, num_classes=train_data.num_classes())
                # loss += 1e-4 * F.mse_loss(logits, y_one_hot.float())

            # TODO: Try adding distance penalty for ordinality?
            # y_one_hot = F.one_hot(y, num_classes=train_data.num_classes())
            # # preds = torch.argmax(model(X), dim=-1)
            # pred_distance = torch.dist(logits, y_one_hot.float(), p=2)
            #
            # loss += 1e-4 * pred_distance

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

            if model_type == 'regression':
                preds = model(X).squeeze()
            else:
                preds = torch.argmax(model(X), dim=-1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(y.tolist())

    # TODO: Paper used MSE not accuracy
    if model_type == 'regression':
        print(f'Test MSE: {mean_squared_error(y_true, y_pred):.4f}')
    else:
        print(y_pred)
        print(y_true)
        missed_pred, missed_true = [], []
        for y_p, y_t in zip(y_pred, y_true):
            if y_p != y_t:
                missed_pred.append(y_p)
                missed_true.append(y_t)
        print(f'Test Accuracy: {accuracy_score(y_true, y_pred):.4f}')
        print(f'Test MSE: {mean_squared_error(y_true, y_pred):.4f}')
        print(f'Test MAE: {mean_absolute_error(y_true, y_pred):.4f}')
        print(f'Test Incorrect MAE: {mean_absolute_error(missed_true, missed_pred):.4f}')

    torch.save(model.state_dict(), f'models/{model_name}.pt')


def ordinalize(x, n_classes=5):
    out = np.zeros(n_classes)
    for i in range(x):
        out[i] = 1
    return out


if __name__ == '__main__':
    main()
