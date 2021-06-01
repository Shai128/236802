import sys
import copy
from scipy import stats
import torch
import numpy as np
import torch.nn as nn
from cqr import helper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


###############################################################################
# Helper functions
###############################################################################

def epoch_internal_train(model, loss_func, x_train, y_train, batch_size, optimizer, cnt=0, best_cnt=np.Inf,
                         require_independence=False):
    """ Sweep over the data and update the model's parameters

    Parameters
    ----------

    model : class of neural net model
    loss_func : class of loss function
    x_train : pytorch tensor n training features, each of dimension p (nXp)
    batch_size : integer, size of the mini-batch
    optimizer : class of SGD solver
    cnt : integer, counting the gradient steps
    best_cnt: integer, stop the training if current cnt > best_cnt

    Returns
    -------

    epoch_loss : mean loss value
    cnt : integer, cumulative number of gradient steps

    """

    model.train()
    shuffle_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    epoch_losses = []
    dependency_losses = []
    pinball_losses = []
    coverage_losses = []
    total_losses = []
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
        if type(loss_func) == CoverageDependenceLoss:
            dependency_losses += [loss_func.dependence_loss]
            pinball_losses += [loss_func.pinball_loss]
            coverage_losses += [loss_func.coverage_loss]
        epoch_losses.append(loss.cpu().detach().numpy())

        if cnt >= best_cnt:
            break

    epoch_loss = np.mean(epoch_losses)
    if require_independence:
        return epoch_loss, cnt, np.mean(dependency_losses), np.mean(pinball_losses), np.mean(coverage_losses)
    else:
        return epoch_loss, cnt


def rearrange(all_quantiles, quantile_low, quantile_high, test_preds):
    """ Produce monotonic quantiles

    Parameters
    ----------

    all_quantiles : numpy array (q), grid of quantile levels in the range (0,1)
    quantile_low : float, desired low quantile in the range (0,1)
    quantile_high : float, desired high quantile in the range (0,1)
    test_preds : numpy array of predicted quantile (nXq)

    Returns
    -------

    q_fixed : numpy array (nX2), containing the rearranged estimates of the
              desired low and high quantile

    References
    ----------
    .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
            "Quantile and probability curves without crossing."
            Econometrica 78.3 (2010): 1093-1125.

    """
    scaling = all_quantiles[-1] - all_quantiles[0]
    low_val = (quantile_low - all_quantiles[0]) / scaling
    high_val = (quantile_high - all_quantiles[0]) / scaling
    q_fixed = np.quantile(test_preds, (low_val, high_val), interpolation='linear', axis=1)
    return q_fixed.T


###############################################################################
# Deep conditional mean regression
# Minimizing MSE loss
###############################################################################

# Define the network
class mse_model(nn.Module):
    """ Conditional mean estimator, formulated as neural net
    """

    def __init__(self,
                 in_shape=1,
                 hidden_size=64,
                 dropout=0.5):
        """ Initialization

        Parameters
        ----------

        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """

        super().__init__()
        self.in_shape = in_shape
        self.out_shape = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return torch.squeeze(self.base_model(x))


# Define the training procedure
class LearnerOptimized:
    """ Fit a neural network (conditional mean) to training data
    """

    def __init__(self, model, optimizer_class, loss_func, device='cpu', test_ratio=0.2, random_state=0):
        """ Initialization

        Parameters
        ----------

        model : class of neural network model
        optimizer_class : class of SGD optimizer (e.g. Adam)
        loss_func : loss to minimize
        device : string, "cuda:0" or "cpu"
        test_ratio : float, test size used in cross-validation (CV)
        random_state : int, seed to be used in CV when splitting to train-test

        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(device)
        self.device = device
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

    def fit(self, x, y, epochs, batch_size, verbose=False):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array, containing the training features (nXp)
        y : numpy array, containing the training labels (n)
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD

        """

        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(device)
        optimizer = self.optimizer_class(model.parameters())
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(x, y, test_size=self.test_ratio, random_state=self.random_state)

        x_train = torch.from_numpy(x_train).float().to(self.device).requires_grad_(False)
        xx = torch.from_numpy(xx).float().to(self.device).requires_grad_(False)
        y_train = torch.from_numpy(y_train).float().to(self.device).requires_grad_(False)
        yy = torch.from_numpy(yy).float().to(self.device).requires_grad_(False)

        best_cnt = 1e10
        best_test_epoch_loss = 1e10

        cnt = 0
        for e in range(epochs):
            epoch_loss, cnt = epoch_internal_train(model, self.loss_func, x_train, y_train, batch_size, optimizer, cnt)
            self.loss_history.append(epoch_loss)

            # test
            model.eval()
            preds = model(xx)
            test_preds = preds.cpu().detach().numpy()
            test_preds = np.squeeze(test_preds)
            test_epoch_loss = self.loss_func(preds, yy).cpu().detach().numpy()

            self.test_loss_history.append(test_epoch_loss)

            if (test_epoch_loss <= best_test_epoch_loss):
                best_test_epoch_loss = test_epoch_loss
                best_epoch = e
                best_cnt = cnt

            if (e + 1) % 100 == 0 and verbose:
                print("CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best loss {}".format(e + 1, epoch_loss,
                                                                                            test_epoch_loss, best_epoch,
                                                                                            best_test_epoch_loss))
                sys.stdout.flush()

        # use all the data to train the model, for best_cnt steps
        x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        cnt = 0
        for e in range(best_epoch + 1):
            if cnt > best_cnt:
                break

            epoch_loss, cnt = epoch_internal_train(self.model, self.loss_func, x, y, batch_size, self.optimizer, cnt,
                                                   best_cnt)
            self.full_loss_history.append(epoch_loss)

            if (e + 1) % 100 == 0 and verbose:
                print("Full: Epoch {}: {}, cnt {}".format(e + 1, epoch_loss, cnt))
                sys.stdout.flush()

    def predict(self, x):
        """ Estimate the label given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of predicted labels (n)

        """
        self.model.eval()
        ret_val = self.model(torch.from_numpy(x).float().to(self.device).requires_grad_(False)).cpu().detach().numpy()
        return ret_val


##############################################################################
# Quantile regression
# Implementation inspired by:
# https://github.com/ceshine/quantile-regression-tensorflow
##############################################################################

class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """

    def __init__(self, quantiles):
        """ Initialize

        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)


        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """

        Parameters
        ----------
        source - 2d matrix
        target - 2d matrix, same shape as source

        Returns
        -------
        MMD Loss of source,target
        """
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class CoverageDependenceLoss(AllQuantileLoss):
    """ Loss based on the dependence of the coverage and average interval size
    """

    def __init__(self, quantiles, min_bin_size, dependence_loss_args=None):
        """ Initialize
        """
        super().__init__(quantiles)
        if dependence_loss_args is None:
            dependence_loss_args = dict()
        self.min_bin_size = min_bin_size
        self.desired_accuracy = max(quantiles) - min(quantiles)

        if dependence_loss_args is None:
            self.dependence_loss_args = {}
        else:
            self.dependence_loss_args = dependence_loss_args
            if 'coverage_loss_multiplier' in dependence_loss_args:
                self.coverage_loss_multiplier = dependence_loss_args.pop('coverage_loss_multiplier')
            else:
                self.coverage_loss_multiplier = 0

    def forward(self, preds, target):
        """ Compute the pinball loss + dependence_loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """

        """
        Alternative pinball loss:
        preds = torch.squeeze(preds)
        y_lower = preds[:, 0]
        y_upper = preds[:, 1]
        relu = torch.nn.ReLU()
        is_beyond_lower = relu(target - y_lower)
        is_below_upper = relu(y_upper - target)
        is_in_interval = is_beyond_lower * is_below_upper
        epsilon = 1e-10
        is_in_interval = is_in_interval / (is_in_interval + epsilon)
        pinball_loss = (is_in_interval - 0.9).norm() / len(preds) + (is_below_upper - is_beyond_lower ).norm() / len(preds) 
        
        """

        pinball_loss = super().forward(preds, target)


        preds = torch.squeeze(preds)
        y_lower = preds[:, 0]
        y_upper = preds[:, 1]

        coverage, avg_interval_len = helper.compute_coverages_and_avg_interval_len(target, y_lower, y_upper)
        coverage_epsilon = 1e-5
        if abs(coverage.mean() - 0) <= coverage_epsilon or abs(coverage.mean() - 1) <= coverage_epsilon:

            coverage_loss = dependence_loss = 0
        else:
            dependence_loss = helper.coverage_and_size_interval_dependence_loss(coverage, avg_interval_len,
                                                                                desired_accuracy=self.desired_accuracy,
                                                                                **self.dependence_loss_args)

            if self.coverage_loss_multiplier != 0:
                coverage_loss = ((coverage.mean() - self.desired_accuracy) ** 2) * self.coverage_loss_multiplier
            else:
                coverage_loss = 0

        self.coverage_loss = (float)(coverage_loss)
        self.pinball_loss = (float)(pinball_loss)
        self.dependence_loss = (float)(dependence_loss)

        loss = pinball_loss + dependence_loss + coverage_loss

        return loss


class all_q_model(nn.Module):
    """ Conditional quantile estimator, formulated as neural net
    """

    def __init__(self,
                 quantiles,
                 in_shape=1,
                 hidden_size=64,
                 dropout=0.5):
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_quantiles),
        )

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return self.base_model(x)


class LearnerOptimizedCrossing:
    """ Fit a neural network (conditional quantile) to training data
    """

    def __init__(self, model, optimizer_class, loss_func, device='cpu', test_ratio=0.2, random_state=0,
                 qlow=0.05, qhigh=0.95, use_rearrangement=False, require_independence=False,
                 iteration_to_start_requiring_independence=20, n_iterations_to_increase_dependence_loss_after=40,
                 dependence_loss_log_initial_reduction=3, y_multiplier=100):
        """ Initialization

        Parameters
        ----------

        model : class of neural network model
        optimizer_class : class of SGD optimizer (e.g. pytorch's Adam)
        loss_func : loss to minimize
        device : string, "cuda:0" or "cpu"
        test_ratio : float, test size used in cross-validation (CV)
        random_state : integer, seed used in CV when splitting to train-test
        qlow : float, low quantile level in the range (0,1)
        qhigh : float, high quantile level in the range (0,1)
        use_rearrangement : boolean, use the rearrangement  algorithm (True)
                            of not (False)

        """
        self.model = model.to(device)
        self.use_rearrangement = use_rearrangement
        self.compute_coverage = True
        self.quantile_low = qlow
        self.quantile_high = qhigh
        self.target_coverage = 100.0 * (self.quantile_high - self.quantile_low)
        self.all_quantiles = loss_func.quantiles
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(device)
        self.device = device
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []
        self.dependence_loss_history = []
        self.pinball_loss_history = []
        self.coverage_loss_history = []
        self.validation_dependency_losses = []
        self.validation_pinball_losses = []
        self.validation_coverage_losses = []
        self.full_dependence_loss_history = []
        self.full_pinball_loss_history = []
        self.full_coverage_loss_history = []
        self.dependence_loss_history_over_train = []

        self.results_during_training = {}

        self.require_independence = require_independence
        self.n_iterations_to_increase_dependence_loss_after = n_iterations_to_increase_dependence_loss_after
        # if require_independence and self.loss_func.dependence_loss_args['cov_loss_multiplier'] < 0.5 and\
        #         self.loss_func.dependence_loss_args['HSIC_loss_multiplier'] < 0.5:
        #     self.require_independence = False

        self.iteration_to_start_requiring_independence = iteration_to_start_requiring_independence
        self.dependence_loss_log_initial_reduction = dependence_loss_log_initial_reduction
        self.y_multiplier = y_multiplier




    def fit(self, x, y, epochs, batch_size, verbose=False):

        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size used in SGD solver

        """


        if self.require_independence:

            self.scaler = MinMaxScaler(feature_range=(-1, abs(max(y)/min(y))))
            # y = y.reshape(len(y), 1)
            # y = self.scaler.fit_transform(y)
            # y = y.reshape(len(y))
            # self.y_multiplier = 100  # 100 / min(abs(y)[y != 0])
            y = y * self.y_multiplier


        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(device)
        optimizer = self.optimizer_class(model.parameters())
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(x,
                                                    y,
                                                    test_size=self.test_ratio,
                                                    random_state=self.random_state)

        x_train = torch.from_numpy(x_train).float().to(self.device).requires_grad_(False)
        xx = torch.from_numpy(xx).float().to(self.device).requires_grad_(False)
        y_train = torch.from_numpy(y_train).float().to(self.device).requires_grad_(False)
        yy_cpu = yy
        yy = torch.from_numpy(yy).float().to(self.device).requires_grad_(False)

        best_avg_length = 1e10
        best_coverage = 0
        best_cnt = 1e10
        best_loss = 1e20
        cnt = 0
        best_pearsons_corr = 2
        achieved_target_coverage = False

        # TODO: delete this line
        n_groups = int(max(x_train[:,0]).item())+1

        if self.require_independence:
            dependence_loss_log_initial_reduction = self.dependence_loss_log_initial_reduction
            n_iterations_to_increase_dependence_loss_after = self.n_iterations_to_increase_dependence_loss_after
            initial_dependence_loss_multiplier = 1
            self.loss_func.dependence_loss_args['total_dependence_multiplier'] = 2**(-dependence_loss_log_initial_reduction)
            minimal_best_epoch = n_iterations_to_increase_dependence_loss_after * dependence_loss_log_initial_reduction \
                                 + self.iteration_to_start_requiring_independence
        else:
            minimal_best_epoch = 0
        for e in range(epochs):
            model.train()
            require_independence = self.require_independence and e > self.iteration_to_start_requiring_independence

            if require_independence:

                if e % n_iterations_to_increase_dependence_loss_after == 0 and\
                        self.loss_func.dependence_loss_args['total_dependence_multiplier']*2 < initial_dependence_loss_multiplier:
                    self.loss_func.dependence_loss_args['total_dependence_multiplier'] *= 2

                elif e % n_iterations_to_increase_dependence_loss_after == 0 and \
                        (self.loss_func.dependence_loss_args['total_dependence_multiplier'] <initial_dependence_loss_multiplier) and\
                        (self.loss_func.dependence_loss_args['total_dependence_multiplier']*2 >= initial_dependence_loss_multiplier):
                    self.loss_func.dependence_loss_args['total_dependence_multiplier'] = initial_dependence_loss_multiplier
                    minimal_best_epoch = e
                    self.minimal_best_epoch = minimal_best_epoch

                epoch_loss, cnt, dependence_loss, pinball_loss, coverage_loss = epoch_internal_train(model, self.loss_func, x_train, y_train,
                                                                        batch_size, optimizer, cnt,
                                                                        require_independence=True)
                self.dependence_loss_history.append(dependence_loss)
                self.pinball_loss_history.append(pinball_loss)
                self.coverage_loss_history.append(coverage_loss)

            else:
                epoch_loss, cnt = epoch_internal_train(model, self.loss_func, x_train, y_train, batch_size, optimizer,
                                                       cnt)
                self.pinball_loss_history.append(epoch_loss)
            self.loss_history.append(epoch_loss)

            with torch.no_grad():
                self.update_results_during_training(model, x_train, y_train, 'train')
                for group_number in range(n_groups):
                    group_idx = (x_train[:, 0] == group_number).cpu().detach().numpy()
                    self.update_results_during_training(model, x_train[group_idx], y_train[group_idx], 'train_group_'+str(group_number))

                if type(self.loss_func) == CoverageDependenceLoss:
                    model.eval()
                    preds = model(x_train)
                    self.loss_func(preds, y_train).cpu().detach().numpy()
                    self.dependence_loss_history_over_train += [self.loss_func.dependence_loss]

                model.eval()
                preds = model(xx)
                test_epoch_loss = self.loss_func(preds, yy).cpu().detach().numpy()
                # print("loss: ", test_epoch_loss)
                if type(self.loss_func) == CoverageDependenceLoss:
                    self.validation_dependency_losses += [self.loss_func.dependence_loss]
                    # print("dependence loss: ", self.loss_func.dependence_loss)
                    self.validation_pinball_losses += [self.loss_func.pinball_loss]
                    self.validation_coverage_losses.append(self.loss_func.coverage_loss)
                else:
                    self.validation_pinball_losses += [test_epoch_loss]

                self.test_loss_history.append(test_epoch_loss)

                test_preds = preds.cpu().detach().numpy()
                test_preds = np.squeeze(test_preds)

                if self.use_rearrangement:
                    test_preds = rearrange(self.all_quantiles, self.quantile_low, self.quantile_high, test_preds)

                y_lower = test_preds[:, 0]
                y_upper = test_preds[:, 1]

                self.update_results_during_training(model, xx, yy_cpu, 'validation')
                for group_number in range(n_groups):
                    group_idx = (xx[:, 0] == group_number).cpu().detach().numpy()
                    self.update_results_during_training(model, xx[group_idx], yy_cpu[group_idx], 'validation_group_'+str(group_number))

                coverage, avg_length = helper.compute_coverage_len(yy_cpu, y_lower, y_upper)

                in_the_range = ((yy_cpu >= y_lower) & (yy_cpu <= y_upper))
                lengths = (y_upper - y_lower)
                corr = stats.pearsonr(in_the_range, lengths)[0]

            if not achieved_target_coverage and coverage > best_coverage:
                achieved_target_coverage = coverage >= self.target_coverage
                best_loss = test_epoch_loss
                best_avg_length = avg_length
                best_coverage = coverage
                best_epoch = e
                best_cnt = cnt


            if ((coverage >= self.target_coverage) and (test_epoch_loss < best_loss) \
                    and (e>minimal_best_epoch) ):
                best_loss = test_epoch_loss
                best_avg_length = avg_length
                best_coverage = coverage
                best_epoch = e
                best_cnt = cnt
                best_pearsons_corr = corr

            if (e + 1) % 100 == 0 and verbose:
                print(
                    "CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best Coverage {} Best Length {} Cur Coverage {}".format(
                        e + 1, epoch_loss, test_epoch_loss, best_epoch, best_coverage, best_avg_length, coverage))
                sys.stdout.flush()

        x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        cnt = 0
        if self.require_independence:
            self.loss_func.dependence_loss_args['total_dependence_multiplier'] = \
                initial_dependence_loss_multiplier / (2 ** dependence_loss_log_initial_reduction)

        self.best_epoch = best_epoch
        y_cpu = y.cpu().detach().numpy()

        for e in range(best_epoch + 1):
            require_independence = self.require_independence and e > self.iteration_to_start_requiring_independence

            if cnt > best_cnt:
                break

            if require_independence:
                if e % n_iterations_to_increase_dependence_loss_after == 0 and\
                        self.loss_func.dependence_loss_args['total_dependence_multiplier']*2 < initial_dependence_loss_multiplier:
                    self.loss_func.dependence_loss_args['total_dependence_multiplier'] *= 2

                elif e % n_iterations_to_increase_dependence_loss_after == 0 and\
                        (self.loss_func.dependence_loss_args['total_dependence_multiplier'] <initial_dependence_loss_multiplier) and\
                        (self.loss_func.dependence_loss_args['total_dependence_multiplier']*2 >= initial_dependence_loss_multiplier):
                    self.loss_func.dependence_loss_args['total_dependence_multiplier'] = initial_dependence_loss_multiplier

                epoch_loss, cnt, dependence_loss, pinball_loss, coverage_loss = epoch_internal_train(self.model, self.loss_func, x, y,
                                                                                      batch_size, self.optimizer, cnt,
                                                                                    best_cnt, require_independence=True)
                self.full_dependence_loss_history.append(dependence_loss)
                self.full_pinball_loss_history.append(pinball_loss)
                self.full_coverage_loss_history.append(coverage_loss)


            else:
                epoch_loss, cnt = epoch_internal_train(self.model, self.loss_func, x, y, batch_size, self.optimizer, cnt,
                                                   best_cnt)

                self.full_pinball_loss_history.append(epoch_loss)
                self.full_loss_history.append(epoch_loss)

            with torch.no_grad():

                self.update_results_during_training(self.model, x, y_cpu, 'final_train')
                for group_number in range(n_groups):
                    group_idx = (x[:, 0] == group_number).cpu().detach().numpy()
                    self.update_results_during_training(self.model, x[group_idx], y_cpu[group_idx], 'final_train_group_'+str(group_number))


            if (e + 1) % 100 == 0 and verbose:
                print("Full: Epoch {}: {}, cnt {}".format(e + 1, epoch_loss, cnt))
                sys.stdout.flush()

    def predict(self, x):
        """ Estimate the conditional low and high quantile given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        test_preds : numpy array of predicted low and high quantiles (nX2)

        """
        self.model.eval()
        test_preds = self.model(torch.from_numpy(x).float().to(self.device).requires_grad_(False)).cpu().detach().numpy()
        if self.use_rearrangement:
            test_preds = rearrange(self.all_quantiles, self.quantile_low, self.quantile_high, test_preds)
        else:
            test_preds[:, 0] = np.min(test_preds, axis=1)
            test_preds[:, 1] = np.max(test_preds, axis=1)

        if self.require_independence:
            test_preds = test_preds / self.y_multiplier
            # test_preds = self.scaler.inverse_transform(test_preds)

        return test_preds


    def update_results_during_training(self, model, x, y, set_name: str):
        if len(x) == 0 or len(y) == 0:
            return
        model.eval()
        idx = np.random.permutation(len(x))  # [:len(xx)]
        preds = model(x[idx])
        test_preds = preds.cpu().detach().numpy()
        test_preds = np.squeeze(test_preds)

        y_lower = test_preds[:, 0]
        y_upper = test_preds[:, 1]
        if torch.is_tensor(y):
            curr_y = y.cpu().detach().numpy()[idx]
        else:
            curr_y = y[idx]
        in_the_range = ((curr_y >= y_lower) & (curr_y <= y_upper))
        lengths = (y_upper - y_lower)

        if 'pearsons_correlation'+'_over_'+set_name not in self.results_during_training:
            self.results_during_training['pearsons_correlation'+'_over_'+set_name] = []

        self.results_during_training['pearsons_correlation'+'_over_'+set_name] += [stats.pearsonr(in_the_range, lengths)[0]]

        if 'coverage'+'_over_'+set_name not in self.results_during_training:
            self.results_during_training['coverage'+'_over_'+set_name] = []

        self.results_during_training['coverage'+'_over_'+set_name] += [np.mean(in_the_range)]

        if 'interval_lengths'+'_over_'+set_name not in self.results_during_training:
            self.results_during_training['interval_lengths'+'_over_'+set_name] = []

        self.results_during_training['interval_lengths'+'_over_'+set_name] += [np.mean(lengths)]



class ApproximateTwoFunctions(nn.Module):
    """ Approximate two functions: f,g for two variables, according to a given loss.
        There are no labels and no test set, we are trying to solve an optimization problem.
    """

    def __init__(self,
                 X_in_features: int,
                 Y_in_features: int,
                 X_hidden_dim: int,
                 Y_hidden_dim: int,
                 loss=None,
                 batch_size=256,
                 dropout=0.1,
                 epochs=500):
        dropout = 0.0
        super(ApproximateTwoFunctions, self).__init__()
        self.loss = loss

        f_layers = [
            nn.Linear(X_in_features, X_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(X_hidden_dim, X_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(X_hidden_dim, X_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(X_hidden_dim, 1),
            nn.Tanh(),
        ]

        g_layers = [
            nn.Linear(Y_in_features, Y_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(Y_hidden_dim, Y_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(Y_hidden_dim, Y_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(Y_hidden_dim, 1),
            nn.Tanh(),
        ]

        self.f_func = nn.Sequential(*f_layers).to(device)
        self.g_func = nn.Sequential(*g_layers).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.epochs = epochs
        self.batch_size = batch_size

    def forward(self, X, Y):
        return self.f_func(X.to(device)), self.f_func(Y.to(device))

    def fit(self, x, y, loss_func=None, epochs=None):

        assert loss_func is not None or self.loss is not None

        if loss_func is None:
            loss_func = self.loss
        if epochs is None:
            epochs = self.epochs
        data_len = x.shape[0]
        batch_size = self.batch_size
        shuffle_idx = np.arange(data_len)
        np.random.shuffle(shuffle_idx)
        x = x[shuffle_idx].detach().to(device)
        y = y[shuffle_idx].detach().to(device)

        self.losses = []

        for _ in range(epochs):
            curr_losses = []

            for idx in range(0, data_len, self.batch_size):

                self.optimizer.zero_grad()
                batch_x = x[idx: min(idx + batch_size, x.shape[0])]
                batch_y = y[idx: min(idx + batch_size, y.shape[0])]
                batch_x.requires_grad = True
                batch_y.requires_grad = True

                x_preds, y_preds = self.forward(batch_x, batch_y)
                loss = loss_func(x_preds, y_preds)
                loss.backward()
                self.optimizer.step()
                curr_losses += [loss.detach().cpu().numpy()]

            curr_losses = np.mean(curr_losses)

            self.losses += [curr_losses]

