
from models import *
# from compas_input import CompasInput
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.transforms import transforms


class DataSet(object):
    def __init__(self):
        pass


class Compas(DataSet):

    def __init__(self, ):
        super(DataSet, self).__init__()
        self.features = [
        "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
        "age", "c_charge_degree", "c_charge_desc", "age_cat", "sex", "race",
        "is_recid"
    ]
        self.features_to_transform = [ "c_charge_degree", "c_charge_desc", "age_cat", "sex", "race",
        "is_recid"
    ]
        self.possible_protected_features = [ 'age_cat', 'sex', 'race']
        self.target_feature_name = 'is_recid'
        self.folder_name = 'compas'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # input = CompasInput(dataset_base_dir='./')
    dataset = Compas()

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


    train = df.iloc[:len(train)]
    test = df.iloc[len(train):]

    protected_feature = dataset.possible_protected_features[0]  # 'age_cat', 'sex', 'race',
    target_name = dataset.target_feature_name

    train_private_feature_values = train[protected_feature]
    train = train.drop([protected_feature], axis=1)

    test_private_feature_values = test[protected_feature]
    test = test.drop([protected_feature], axis=1)

    train_y, test_y = train[target_name], test[target_name]

    train_x, test_x = train.drop([target_name], axis=1), test.drop([target_name],axis=1)

    adversarial_model = AdversarialReweightedModel(in_features=train_x.shape[1],
                                                  hidden_dim1=32,
                                                  hidden_dim2=64,
                                                  dropout=0.1,
                                                  lr=1e-4,
                                                  use_reweighting=True,
                                                  device='cpu')

    adversarial_model.fit(torch.tensor(train_x.values, dtype=torch.float32),
                         torch.tensor(train_y.values, dtype=torch.float32), epochs=200)

    plt.plot(adversarial_model.learner_losses)
    plt.title("Learner Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(adversarial_model.adversarial_losses)
    plt.title("Adversarial Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    train_preds = adversarial_model.predict(torch.tensor(train_x.values, dtype=torch.float32))
    train_y = torch.tensor(train_y.values, dtype=torch.int64)
    train_accuracy = (train_preds == train_y).float().mean()

    test_preds = adversarial_model.predict(torch.tensor(test_x.values, dtype=torch.float32))
    test_y = torch.tensor(test_y.values, dtype=torch.int64)
    test_accuracy = (test_preds ==test_y).float().mean()


    print(f"train acc: {train_accuracy}")
    print(f"test acc: {test_accuracy}")

    for protected_feature_value in range(max(test_private_feature_values)+1):
        idx = test_private_feature_values==protected_feature_value
        print(f"test protected feature value= {protected_feature_value} accuracy: {(test_preds[idx]== test_y[idx]).float().mean()}")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
