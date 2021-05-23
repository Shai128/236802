import numpy as np
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from utils import pearsons_corr, HSIC, pearsons_corr_2d


def change_parameters_require_grad(parameters, require_grad):
    for param in parameters:
        param.requires_grad = require_grad


class BaseModel(nn.Module):
    model_name: str = 'baseline'

    def __init__(self,
                 in_features: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 dropout=0.1,
                 lr=1e-3,
                 take_best_model=False,
                 wait=50,
                 device='cpu'):

        super(BaseModel, self).__init__()

        learner_layers = [
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 2),

        ]
        self.wait = wait
        self.take_best_model = take_best_model
        self.learner = nn.Sequential(*learner_layers).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        self.device = device

    def calculate_loss(self, x, y):
        return self.cross_entropy(self.learner_forward(x), y).mean()

    def learner_forward(self, x):
        return self.learner(x)

    def get_scores(self, x):
        x_device = x.device
        x = x.to(self.device)

        return torch.softmax(self.learner(x), dim=1)[:, 1].to(x_device)

    def predict(self, x):
        x_device = x.device
        x = x.to(self.device)
        return torch.argmax(self.learner_forward(x), dim=1).to(x_device)

    def fit(self, x, y, val_x=None, val_y=None, epochs=500, batch_size=64):

        device = self.device

        x = x.to(device)
        # y = y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        y = y.type(torch.LongTensor).to(device)

        loader = DataLoader(TensorDataset(x, y),
                            shuffle=True,
                            batch_size=batch_size)

        self.losses = []
        self.learner_losses = []
        self.best_loss = None
        self.epochs_not_improved = 0
        self.val_loss = []

        for e in (range(epochs)):
            curr_losses = []
            curr_learner_loss = []
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                batch_x.requires_grad = True
                batch_y.requires_grad = False
                # learner backward
                change_parameters_require_grad(self.learner.parameters(), require_grad=True)

                learner_loss = self.calculate_loss(batch_x, batch_y)
                learner_loss.backward()

                loss = learner_loss

                self.optimizer.step()
                curr_losses += [loss.detach().cpu().numpy()]
                curr_learner_loss += [learner_loss.detach().cpu().numpy()]

            curr_losses = np.mean(curr_losses)
            curr_learner_loss = np.mean(curr_learner_loss)

            self.losses += [curr_losses]
            self.learner_losses += [curr_learner_loss]

            if self.early_stop(val_x, val_y):
                print(f"finished at epoch {e}")
                break

    def early_stop(self, val_x, val_y):
        with torch.no_grad():
            if val_x is not None and val_y is not None:
                learner_loss = self.calculate_loss(val_x, val_y)
                self.val_loss += [learner_loss]

                if self.best_loss is None or learner_loss < self.best_loss:
                    self.epochs_not_improved = 0
                    self.best_loss = learner_loss
                    if self.take_best_model:
                        self.best_learner = deepcopy(self.learner)
                else:
                    self.epochs_not_improved += 1
                    if self.epochs_not_improved >= self.wait:
                        if self.take_best_model:
                            self.learner = self.best_learner
                        return True
            return False

    def plot_loss(self):
        plt.plot(self.learner_losses)
        plt.title("Train: Learner Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.plot(self.val_loss)
        plt.title("Validation: Learner Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


class AdversarialReweightedModel(BaseModel):
    model_name: str = 'ARL'

    def __init__(self,
                 in_features: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 dropout=0.1,
                 lr=1e-3,
                 device='cpu', **kw):

        super(AdversarialReweightedModel, self).__init__(in_features, hidden_dim1, hidden_dim2, dropout,
                                                         lr=lr, device=device, **kw)

        adversary_layers = [
            nn.Linear(in_features + 1, 1),
            torch.nn.Sigmoid()
        ]

        self.adversary = nn.Sequential(*adversary_layers).to(device)

        self.adversary_optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = device

    def adversary_forward(self, x, y):
        f_phi = self.adversary(torch.cat([x, y.reshape((len(y), 1))], dim=1)).flatten()
        lambda_phi = 1 + len(x) * (f_phi / torch.sum(f_phi))
        return lambda_phi

    def fit(self, x, y, val_x=None, val_y=None, epochs=500, batch_size=64):

        device = self.device

        x = x.to(device)
        # y = y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        y = y.float().to(device)

        loader = DataLoader(TensorDataset(x, y),
                            shuffle=True,
                            batch_size=batch_size)

        self.best_loss = None
        self.epochs_not_improved = 0

        self.losses = []
        self.learner_losses = []
        self.adversarial_losses = []
        self.val_loss = []

        cross_entropy = self.cross_entropy

        for _ in (range(epochs)):
            curr_losses = []
            curr_learner_loss = []
            curr_adversarial_loss = []
            for batch_x, batch_y in loader:
                batch_x.requires_grad = True

                # adversary backward
                self.adversary_optimizer.zero_grad()
                change_parameters_require_grad(self.adversary.parameters(), require_grad=True)
                change_parameters_require_grad(self.learner.parameters(), require_grad=False)

                batch_y.requires_grad = True
                lambdas = self.adversary_forward(batch_x, batch_y)
                y_cross_entropy = batch_y.type(torch.LongTensor).clone().to(device)
                y_cross_entropy.requires_grad = False
                adversary_loss = - lambdas @ cross_entropy(self.learner_forward(batch_x.detach()), y_cross_entropy)
                adversary_loss.backward()
                self.adversary_optimizer.step()

                # learner backward
                self.optimizer.zero_grad()
                change_parameters_require_grad(self.adversary.parameters(), require_grad=False)
                change_parameters_require_grad(self.learner.parameters(), require_grad=True)
                lambdas = self.adversary_forward(batch_x.detach(), batch_y.detach())
                y_cross_entropy = batch_y.type(torch.LongTensor).clone().to(device)
                y_cross_entropy.requires_grad = False
                learner_loss = lambdas @ cross_entropy(self.learner_forward(batch_x), y_cross_entropy)

                learner_loss.backward()

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

            if self.early_stop(val_x, val_y):
                break

    def plot_loss(self):
        super(AdversarialReweightedModel, self).plot_loss()
        plt.plot(self.adversarial_losses)
        plt.title("Adversarial Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


class ImprovedModel(BaseModel):
    model_name: str = 'ImprovedModel'

    def __init__(self,
                 in_features: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 dropout=0.1,
                 lr=1e-3,
                 device='cpu',
                 decorr_mult=0.1,
                 discriminator_mult=0.1,
                 **kw):
        super(ImprovedModel, self).__init__(in_features, hidden_dim1, hidden_dim2, dropout,
                                            lr=lr, device=device, **kw)
        adversary_layers = [
            nn.Linear(in_features + 1, 32, bias=True),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(32, 32, bias=True),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(32, 1, bias=True),

        ]
        self.adversary_network = nn.Sequential(*adversary_layers).to(device)
        self.adversary_optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.decorr_mult = decorr_mult
        self.discriminator_mult = discriminator_mult

    def calculate_loss(self, x, y, return_losses=False):
        learner_loss = self.cross_entropy(self.learner_forward(x), y)

        indp_loss = self.decorr_mult *\
                    torch.abs(pearsons_corr_2d(torch.cat([x, y.reshape(len(y), 1)], dim=1), learner_loss)).mean()

        discriminating_loss = -self.discriminator_mult * (1. / learner_loss.mean().detach()) * \
                              (learner_loss - self.adversary_network(
                                  torch.cat([x, y.reshape(len(y), 1)], dim=1)).detach().reshape(len(x))).abs().mean()

        loss = learner_loss.mean() + indp_loss + discriminating_loss
        if return_losses:
            return loss, learner_loss, indp_loss
        else:
            return loss

    def calc_adversary_loss(self, x, y):
        learner_loss = self.cross_entropy(self.learner_forward(x), y)
        discriminating_loss = (
                    learner_loss - self.adversary_network(torch.cat([x, y.reshape(len(y), 1)], dim=1)).detach().reshape(
                len(x))).abs().mean()

        return discriminating_loss

    def fit(self, x, y, val_x=None, val_y=None, epochs=500, batch_size=64):
        device = self.device

        x = x.to(device)
        # y = y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        y = y.type(torch.LongTensor).to(device)

        loader = DataLoader(TensorDataset(x, y),
                            shuffle=True,
                            batch_size=batch_size)

        self.losses = []
        self.learner_losses = []
        self.val_loss = []

        self.epochs_not_improved = 0
        self.best_loss = None
        self.adversary_losses = []

        for e in (range(epochs)):
            curr_losses = []
            curr_learner_loss = []
            curr_adversary_losses = []
            epoch_x, epoch_y, epoch_losses = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)
            for batch_x, batch_y in loader:
                batch_x.requires_grad = True
                batch_y.requires_grad = False

                # adversary backward
                self.adversary_optimizer.zero_grad()
                change_parameters_require_grad(self.learner.parameters(), require_grad=False)
                change_parameters_require_grad(self.adversary_network.parameters(), require_grad=True)
                adversary_loss = self.calc_adversary_loss(batch_x, batch_y)
                adversary_loss.backward()
                self.adversary_optimizer.step()
                # adversary_loss = torch.Tensor([0])

                # learner backward
                self.optimizer.zero_grad()
                change_parameters_require_grad(self.learner.parameters(), require_grad=True)
                change_parameters_require_grad(self.adversary_network.parameters(), require_grad=False)
                loss, learner_loss, indp_loss = self.calculate_loss(batch_x, batch_y, return_losses=True)
                loss.backward()
                self.optimizer.step()

                epoch_x = torch.cat([epoch_x, batch_x.detach()], dim=0)
                epoch_y = torch.cat([epoch_y, batch_y.detach()], dim=0)
                epoch_losses = torch.cat([epoch_losses, learner_loss.detach()], dim=0)

                learner_loss = learner_loss.mean()
                curr_losses += [loss.detach().cpu().numpy()]
                curr_learner_loss += [learner_loss.detach().cpu().numpy()]
                curr_adversary_losses += [adversary_loss.detach().cpu().numpy()]


            curr_losses = np.mean(curr_losses)
            curr_learner_loss = np.mean(curr_learner_loss)
            curr_adversary_losses = np.mean(curr_adversary_losses)

            self.losses += [curr_losses]
            self.learner_losses += [curr_learner_loss]
            self.adversary_losses += [curr_adversary_losses]

            if self.early_stop(val_x, val_y):
                # print(f"finished at epoch {e}")
                break

    def plot_loss(self):
        super(ImprovedModel, self).plot_loss()
        plt.plot(self.adversary_losses)
        plt.title("Train: Adversary Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
