#!/usr/bin/env python
# coding: utf-8

# Let's try to refactor the code and parameterized the half-life predictor with pytorch.

# ## Variations of Half-life regression (HLR)
#
# short-hand for each record \begin{align}<\cdot>&=<\Delta,x,P[\text{recall}]\in[0,1]>\\&=<\Delta,x,y\in\{0,1\}>\end{align}
#
# Regression against recall probability $$l_\text{recall}(<\cdot>;\theta)=(p-f_\theta(x,\Delta))^2$$
#
# Regression against back-solved half-life $$l_\text{half-life}(<\cdot>;\theta)=(\frac{-\Delta}{\log_2{p}}-f_\theta(x,\Delta))^2$$
#
# Binary recall classification $$l_\text{binary}(<\cdot>;\theta)=\text{xent}(f_\theta(x,\Delta),y)$$
#
# Assume that half-life increases exponentially with each repeated exposure, with a linear approximator, you get $f_\theta(x,\Delta)=2^{\theta\cdot x}$. Use this parameterization with regression against both recall probability and back-solved half-life, you get Settles' formulation:
# $$l(<\cdot>; \theta)=(p-2^{\frac{\Delta}{2^{\theta\cdot x}}})^2+\alpha(\frac{\Delta}{\log_2(p)}-2^{\theta\cdot{x}})^2+\lambda|\theta|_2^2$$


import os
import math
import pickle
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


MIN_HALF_LIFE = 15.0 / (24 * 60)    # 15 minutes
MAX_HALF_LIFE = 274.                # 9 months
LN2 = math.log(2)


def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)


def hclip(h):
    # bound min/max half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def featurize(df):
    # disable feature by removing prefix f_
    df.p_recall = df.p_recall.apply(pclip)
    df.delta = df.delta.apply(lambda x: x / (60 * 60 * 24))  # in days
    df['half_life'] = df.apply(lambda x: hclip(- x['delta'] / (math.log(x['p_recall'], 2))), axis=1)
    # df['f_lang'] = df.apply(lambda x: '%s->%s' % (x['ui_language'], x['learning_language']), axis=1)
    df['f_history_correct_sqrt'] = df.history_correct.apply(lambda x: math.sqrt(1 + x))
    df['f_history_wrong_sqrt'] = df.apply(lambda x: math.sqrt(1 + x['history_seen'] - x['history_correct']), axis=1)
    # df = df.rename({'delta': 'f_delta'}, axis=1)
    df = df.rename({'history_seen': 'f_history_seen'}, axis=1)
    df = df.rename({'history_correct': 'f_history_correct'}, axis=1)
    # df = df.rename({'session_seen': 'f_session_seen'}, axis=1)
    # df = df.rename({'session_correct': 'f_session_correct'}, axis=1)
    df = df.rename({'history_correct_sqrt': 'f_history_correct_sqrt'}, axis=1)
    df = df.rename({'history_wrong_sqrt': 'f_history_wrong_sqrt'}, axis=1)
    df['bias'] = 1
    return df


def get_featurized_df():
    featurized_df_dir = 'data/features.h5'

    if os.path.exists(featurized_df_dir):
        print('loading featurized_df')
        return pd.read_hdf(featurized_df_dir)

    df = pd.read_csv('./data/settles.acl16.learning_traces.13m.csv.gz')
    df = featurize(df)

    df.to_hdf(featurized_df_dir, 'data')

    return df


def get_split_dfs():
    train_df_dir = 'data/train.h5'
    test_df_dir = 'data/test.h5'

    if os.path.exists(train_df_dir) and os.path.exists(test_df_dir):
        print('loading train test df')
        return pd.read_hdf(train_df_dir), pd.read_hdf(test_df_dir)

    df = get_featurized_df()

    splitpoint = int(0.9 * len(df))
    train_df, test_df = df.iloc[:splitpoint], df.iloc[splitpoint:]

    train_df.to_hdf(train_df_dir, 'data')
    test_df.to_hdf(test_df_dir, 'data')

    return train_df, test_df


def get_split_numpy():
    dirs = [
        'data/x_train.npy',
        'data/y_train.npy',
        'data/x_test.npy',
        'data/y_test.npy'
    ]
    if all(os.path.exists(d) for d in dirs):
        print('loading train test numpy')
        return (np.load(d) for d in dirs)

    train_df, test_df = get_split_dfs()

    feature_names = [c for c in train_df.columns if c.startswith('f_')] + ['bias']
    print('features', feature_names)
    x_train = train_df[feature_names].to_numpy().astype(np.float32)
    y_train = train_df['p_recall'].to_numpy().astype(np.float32)
    x_test = test_df[feature_names].to_numpy().astype(np.float32)
    y_test = test_df['p_recall'].to_numpy().astype(np.float32)

    np.save(dirs[0], x_train)
    np.save(dirs[1], y_train)
    np.save(dirs[2], x_test)
    np.save(dirs[3], y_test)

    return x_train, y_train, x_test, y_test


class RetentionDataset(torch.utils.data.Dataset):

    def __init__(self, fold='train'):
        x_train, y_train, x_test, y_test = get_split_numpy()
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)
        self.mean[-1] = 0
        self.std[-1] = 1

        if fold == 'train':
            self.x, self.y = x_train, y_train
        elif fold == 'test':
            self.x, self.y = x_test, y_test

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = (self.x[idx] - self.mean) / self.std
        y = np.array(self.y[idx])
        return torch.from_numpy(x), torch.from_numpy(y)


class Net(nn.Module):

    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(n_input, 128)
        # self.dropout1 = nn.Dropout(0.25)
        # self.fc2 = nn.Linear(128, 2)
        self.fc1 = nn.Linear(n_input, n_output)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # return x
        return self.fc1(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_func = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits[:, 0], target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    loss_func = nn.MSELoss(reduction='mean')
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)[:, 0]
            # sum up batch loss
            test_loss += loss_func(logits, target).item()
            # get the index of the max log-probability
            predictions += logits.detach().cpu().numpy().tolist()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss, predictions


def main():
    parser = argparse.ArgumentParser(description='Retention model')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_dataset = RetentionDataset('train')
    test_dataset = RetentionDataset('test')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    n_input = train_dataset.x.shape[1]
    model = Net(n_input=n_input, n_output=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = "checkpoints/retention_model.pt"
    prediction_dir = "checkpoints/predictions.pkl"

    best_test_loss = 9999
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss, predictions = test(args, model, device, test_loader)
        scheduler.step()
        if test_loss < best_test_loss:
            torch.save(model.state_dict(), checkpoint_dir)
            print('save model checkpoint to', checkpoint_dir)
            with open(prediction_dir, 'wb') as f:
                pickle.dump(predictions, f)
            print('save predictions to', prediction_dir)
            best_test_loss = test_loss


if __name__ == '__main__':
    main()
