import sys
import copy
from scipy import stats
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim

def epoch_internal_train(model, loss_func, x_train, y_train, batch_size, optimizer, cnt=0, best_cnt=np.Inf,
                         ):

    model.train()
    shuffle_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    epoch_losses = []
    for idx in range(0, x_train.shape[0], batch_size):
        cnt = cnt + 1
        optimizer.zero_grad()
        batch_x = x_train[idx: min(idx + batch_size, x_train.shape[0]), :]
        batch_y = y_train[idx: min(idx + batch_size, y_train.shape[0])]
        batch_x.requires_grad = True
        preds = model(batch_x)
        loss = loss_func(preds, batch_y)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.cpu().detach().numpy())

        if cnt >= best_cnt:
            break

    epoch_loss = np.mean(epoch_losses)

    return epoch_loss, cnt


def change_parameters_require_grad(parameters, require_grad):
    for param in parameters:
        param.requires_grad = require_grad


class AdversarialReweightedModel(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 dropout=0.1,
                 lr=1e-3,
                 use_reweighting=True,
                 device='cpu'):

        super(AdversarialReweightedModel, self).__init__()

        adversary_layers = [

            nn.Linear(in_features + 1, 1),
            torch.nn.Sigmoid()
        ]

        learner_layers = [
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 2),

        ]

        self.adversary = nn.Sequential(*adversary_layers).to(device)
        self.learner = nn.Sequential(*learner_layers).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device=device
        self.use_reweighting = use_reweighting

    def adversary_forward(self, x, y):
        f_phi = self.adversary(torch.cat([x, y.reshape((len(y), 1))], dim=1)).flatten()
        lambda_phi = 1 + len(x) * (f_phi / torch.sum(f_phi))
        return lambda_phi

    def learner_forward(self, x):
        return self.learner(x)

    def predict(self, x):
        return torch.argmax(self.learner_forward(x), dim=1)

    def fit(self, x, y, epochs=500, batch_size=64):

        device = self.device

        data_len = x.shape[0]
        batch_size = batch_size
        shuffle_idx = np.arange(data_len)
        np.random.shuffle(shuffle_idx)
        x = x[shuffle_idx].detach().to(device)
        y = y[shuffle_idx].detach().to(device)

        self.losses = []
        self.learner_losses = []
        self.adversarial_losses = []

        cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        pbar_file = sys.stdout
        with tqdm(epochs, file=pbar_file) as pbar:
            for _ in tqdm(range(epochs)):
                curr_losses = []
                curr_learner_loss = []
                curr_adversarial_loss = []
                for idx in range(0, data_len, batch_size):

                    self.optimizer.zero_grad()
                    batch_x = x[idx: min(idx + batch_size, x.shape[0])]
                    batch_y = y[idx: min(idx + batch_size, y.shape[0])]
                    curr_batch_size = batch_x.shape[0]
                    batch_x.requires_grad = True
                    batch_y.requires_grad = False

                    # learner backward
                    change_parameters_require_grad(self.adversary.parameters(), require_grad=False)
                    change_parameters_require_grad(self.learner.parameters(), require_grad=True)

                    if self.use_reweighting:
                        lambdas = self.adversary_forward(batch_x, batch_y)
                    else:
                        lambdas = torch.ones(curr_batch_size) / curr_batch_size
                    y_cross_entropy = batch_y.type(torch.LongTensor).clone()
                    learner_loss = lambdas @ cross_entropy(self.learner_forward(batch_x), y_cross_entropy)

                    learner_loss.backward()

                    # adversary backward
                    change_parameters_require_grad(self.adversary.parameters(), require_grad=True)
                    change_parameters_require_grad(self.learner.parameters(), require_grad=False)

                    if self.use_reweighting:
                        batch_y.requires_grad = True
                        lambdas = self.adversary_forward(batch_x, batch_y)
                        y_cross_entropy = batch_y.type(torch.LongTensor).clone()
                        y_cross_entropy.requires_grad = False
                        adversary_loss = - lambdas @ cross_entropy(self.learner_forward(batch_x), y_cross_entropy)
                        adversary_loss.backward()

                    else:
                        adversary_loss = torch.Tensor([0])

                    loss = learner_loss + adversary_loss

                    self.optimizer.step()
                    curr_losses += [loss.detach().cpu().numpy()]
                    curr_learner_loss += [learner_loss.detach().cpu().numpy()]
                    curr_adversarial_loss += [adversary_loss.detach().cpu().numpy()]

                curr_losses = np.mean(curr_losses)
                curr_learner_loss = np.mean(curr_learner_loss)
                curr_adversarial_loss = np.mean(curr_adversarial_loss)

                self.losses += [curr_losses]
                self.learner_losses += [curr_learner_loss]
                self.adversarial_losses += [curr_adversarial_loss]

                # pbar.set_description(f"learner loss: {curr_learner_loss}, loss: {curr_losses}")
                # pbar.update(n=1)


