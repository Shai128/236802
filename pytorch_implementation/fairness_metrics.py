from sklearn import metrics
import numpy as np

def auc(y, y_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_scores)
    return metrics.auc(fpr, tpr)

def accuracy(y, y_pred):
    return np.mean(y == y_pred)

def calculate_fairness_metrics(test_private_features_values, protected_feature_names, y_test, y_scores):
    y_pred = y_scores > 0.5
    result_dict = {}

    result_dict['Accuracy'] = accuracy(y_test, y_pred)
    result_dict['AUC'] = auc(y_test, y_scores)

    subgroups = np.unique(test_private_features_values.values, axis=0)

    minority_subgroup_name = None
    minority_subgroup_size = None

    for id, subgroup in enumerate(subgroups):
        idx = (test_private_features_values == subgroup).min(axis=1)
        subgroup_name = f'subgroup {id}'
        if minority_subgroup_name is None or len(y_test[idx]) < minority_subgroup_size:
            minority_subgroup_name = subgroup_name
            minority_subgroup_size = len(y_test[idx])

        result_dict[f'Accuracy {subgroup_name}'] = accuracy(y_test[idx], y_pred[idx])
        result_dict[f'AUC {subgroup_name}'] = auc(y_test[idx], y_scores[idx])

    result_dict['minority subgroup AUC'] = result_dict[f'AUC {minority_subgroup_name}']
    AUCs = [result_dict[f'AUC subgroup {id}'] for id in range(len(subgroups))]
    result_dict['min subgroup AUC'] = min(AUCs)
    result_dict['average subgroup AUC'] = np.mean(AUCs)


    for feature in protected_feature_names:
        for id, value in enumerate(np.unique(test_private_features_values[feature])):
            idx = test_private_features_values[feature] == value
            group_name = f"{feature} group {id}"

            result_dict[f'Accuracy {group_name}'] = accuracy(y_test[idx], y_pred[idx])
            result_dict[f'AUC {group_name}'] = auc(y_test[idx], y_scores[idx])


    return result_dict

