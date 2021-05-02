#!/usr/bin/env python3

import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from dataset import BldgDataset
from model import ClassificationModel


def main():
    model_name = 'model_08_cols_w_dist'
    os.makedirs(f'models/{model_name}', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    EPOCHS = 20
    lr = 5e-3
    l2_reg = 0.1
    n_trials = 10

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

    acc_list, mse_list, mae_list, mae_missed_list = [], [], [], []
    for trial in range(n_trials):

        test_df = data.sample(frac=0.2)
        train_df = data.drop(test_df.index)

        train_data = BldgDataset(train_df)
        test_data = BldgDataset(test_df)

        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)
        test_loader = DataLoader(test_data,
                                 batch_size=batch_size*2,
                                 shuffle=False)

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

                # TODO: Find a loss function for ordinal classification. Try adding distance penalty for ordinality?
                loss = F.cross_entropy(logits, y)
                # y_one_hot = F.one_hot(y, num_classes=train_data.num_classes())
                # loss += 1e-4 * F.mse_loss(logits, y_one_hot.float())

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

                preds = torch.argmax(model(X), dim=-1)

                y_pred.extend(preds.cpu().tolist())
                y_true.extend(y.tolist())

        missed_pred, missed_true = [], []
        for y_p, y_t in zip(y_pred, y_true):
            if y_p != y_t:
                missed_pred.append(y_p)
                missed_true.append(y_t)

        acc = accuracy_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mae_missed = mean_absolute_error(missed_true, missed_pred)

        print(f'Test Accuracy: {acc:.4f}')
        print(f'Test MSE: {mse:.4f}')
        print(f'Test MAE: {mae:.4f}')
        print(f'Test Incorrect MAE: {mae_missed:.4f}')

        acc_list.append(acc)
        mse_list.append(mse)
        mae_list.append(mae)
        mae_missed_list.append(mae_missed)

        torch.save(model.state_dict(), f'models/{model_name}/{model_name}_{trial}.pt')

    print('-' * 30)
    print(f'Avg Test Accuracy: {mean(acc_list):.4f} ({stdev(acc_list):.4f})')
    print(f'Avg Test MSE: {mean(mse_list):.4f} ({stdev(mse_list):.4f})')
    print(f'Avg Test MAE: {mean(mae_list):.4f} ({stdev(mae_list):.4f})')
    print(f'Avg Test Incorrect MAE: {mean(mae_missed_list):.4f} ({stdev(mae_missed_list):.4f})')


if __name__ == '__main__':
    main()
