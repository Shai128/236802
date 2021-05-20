import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(dataset):
    data_features = dataset.features

    train = pd.read_csv(f"data/{dataset.folder_name}/train.csv", header=None, names=data_features)
    test = pd.read_csv(f"data/{dataset.folder_name}/test.csv", header=None, names=data_features)

    df = pd.concat([train, test], axis=0)

    features_to_transform = dataset.features_to_transform
    for feature_name in features_to_transform:
        df[feature_name] = pd.Categorical(df[feature_name])
        df[feature_name] = df[feature_name].cat.codes

    for feature_name in data_features:
        df[feature_name] = pd.to_numeric(df[feature_name])

    idx = np.random.permutation(len(df))
    train = df.iloc[idx[:len(train)]]

    test = df.iloc[idx[len(train):]]

    protected_features = dataset.possible_protected_features  # 'sex', 'race',
    target_name = dataset.target_feature_name

    train_private_features_values = train[protected_features]
    train = train.drop(protected_features, axis=1)

    test_private_features_values = test[protected_features]
    test = test.drop(protected_features, axis=1)

    val_ratio = 0.15
    val = train.iloc[:int(val_ratio * len(train))]
    train = train.iloc[int(val_ratio * len(train)):]

    train_y, test_y, val_y = train[target_name], test[target_name], val[target_name]

    train_x, test_x, val_x = train.drop([target_name], axis=1), test.drop([target_name], axis=1), val.drop(
        [target_name], axis=1)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    val_x = scaler.transform(val_x)

    return train_x, test_x, val_x, train_y, test_y, val_y, train_private_features_values, test_private_features_values, protected_features


def pearsons_corr(x, y):

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))



def pearsons_corr_2d(x, y):
    """

    Parameters
    ----------
    x - 2d matrix
    y - 1d vector

    Returns
    -------
    vector of correlations between each feature of x to y
    """
    vx = x - torch.mean(x, dim=0)
    vy = y - torch.mean(y)

    corrs = torch.sum(vx .T* vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2)))
    return corrs


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)



def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x).float()
    L = GaussianKernelMatrix(y, s_y).float()
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().to(K.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC

