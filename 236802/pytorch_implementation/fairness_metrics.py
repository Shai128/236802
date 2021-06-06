import torch
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import pearsons_corr_2d, pearsons_corr, HSIC

def auc(y, y_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_scores)
    return metrics.auc(fpr, tpr)

def accuracy(y, y_pred):
    return np.mean(y == y_pred)

def calculate_fairness_metrics(test_private_features_values, protected_feature_names, x_test, y_test, y_scores):
    y_pred = y_scores > 0.5
    result_dict = {}
    y_test = y_test.values
    result_dict['Accuracy'] = accuracy(y_test, y_pred)
    result_dict['AUC'] = auc(y_test, y_scores)

    subgroups = np.unique(test_private_features_values.values, axis=0)

    minority_subgroup_name = None
    minority_subgroup_size = None

    for id, subgroup in enumerate(subgroups):
        idx = (test_private_features_values == subgroup).min(axis=1)
        if min(y_test[idx]) != 0 or max(y_test[idx]) != 1:
            continue
        subgroup_name = f'subgroup {id}'
        if minority_subgroup_name is None or len(y_test[idx]) < minority_subgroup_size:
            minority_subgroup_name = subgroup_name
            minority_subgroup_size = len(y_test[idx])

        result_dict[f'Accuracy {subgroup_name}'] = accuracy(y_test[idx], y_pred[idx])
        result_dict[f'AUC {subgroup_name}'] = auc(y_test[idx], y_scores[idx])

    result_dict['minority subgroup AUC'] = result_dict[f'AUC {minority_subgroup_name}']
    AUCs = [result_dict[f'AUC subgroup {id}'] for id in range(len(subgroups)) if f'AUC subgroup {id}' in result_dict]
    result_dict['min subgroup AUC'] = min(AUCs)
    result_dict['average subgroup AUC'] = np.mean(AUCs)


    for feature in protected_feature_names:
        for id, value in enumerate(np.unique(test_private_features_values[feature])):
            idx = test_private_features_values[feature] == value
            if min(y_test[idx]) != 0 or max(y_test[idx]) != 1:
                continue
            group_name = f"{feature} group {id}"

            result_dict[f'Accuracy {group_name}'] = accuracy(y_test[idx], y_pred[idx])
            result_dict[f'DAccuracy {group_name}'] = abs(result_dict[f'Accuracy {group_name}'] - result_dict['Accuracy'])
            result_dict[f'AUC {group_name}'] = auc(y_test[idx], y_scores[idx])

    result_dict['corr'] = pearsons_corr_2d(torch.Tensor(x_test), torch.Tensor(y_test == y_pred).float()).abs().mean().item()
    result_dict['HSIC'] = HSIC(torch.Tensor(x_test), torch.Tensor(y_test == y_pred).float().reshape((len(y_test), 1))).abs().mean().item()

    wsc_res = wsc_unbiased(x_test, y_test, y_pred, delta=0.1, M=1000, test_size=0.75,
                                           random_state=2021, verbose=False)
    result_dict['Test WSC'] = wsc_res

    result_dict['Test Delta WSC'] = abs(wsc_res - result_dict['Accuracy'])
    return result_dict


def wsc(X, y, y_pred, delta=0.1, M=1000, verbose=False):
    def wsc_v(X, y, y_pred, delta, v):
        n = len(y)
        cover = (y == y_pred).astype(np.float32)
        z = np.dot(X, v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, y_pred, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, y_pred, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_unbiased(X, y, y_pred, delta=0.1, M=1000, test_size=0.75, random_state=2021, verbose=False):
    # X, y, y_pred = X.numpy(), y.numpy(), y_pred.numpy()

    def wsc_vab(X, y, y_pred, v, a, b):
        pred_correct = (y == y_pred).astype(np.float32)
        z = np.dot(X, v)
        idx = np.where((z >= a) * (z <= b))
        coverage = np.mean(pred_correct[idx])
        return coverage

    X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = \
        train_test_split(X, y, y_pred, test_size=test_size,
                         random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, y_train, y_pred_train, delta=delta, M=M, verbose=verbose)
    # Estimate coverage
    coverage = wsc_vab(X_test, y_test, y_pred_test, v_star, a_star, b_star)
    return coverage

