
from models import *
# from compas_input import CompasInput
import pandas as pd
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # input = CompasInput(dataset_base_dir='./')

    compas_features = [
        "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
        "age", "c_charge_degree", "c_charge_desc", "age_cat", "sex", "race",
        "is_recid"
    ]

    train = pd.read_csv("data/compas/train.csv", header=None, names=compas_features)
    test = pd.read_csv("data/compas/test.csv", header=None, names=compas_features)

    df = pd.concat([train, test], axis=0)

    features_to_transform = [ "c_charge_degree", "c_charge_desc", "age_cat", "sex", "race",
        "is_recid"
    ]
    for feature_name in features_to_transform:
        df[feature_name] = pd.Categorical(df[feature_name])
        df[feature_name] = df[feature_name].cat.codes

    for feature_name in compas_features:
        df[feature_name] = pd.to_numeric(df[feature_name])


    train = df.iloc[:len(train)]
    test = df.iloc[len(train):]

    protected_feature = 'race'
    target_name = 'is_recid'

    private_feature_values = train.drop([protected_feature], axis=1)

    train_y, test_y = train[target_name], test[target_name]

    train_x, test_x = train.drop([target_name], axis=1), test.drop([target_name],axis=1)

    adverarial_model = AdversarialReweightedModel(in_features=train_x.shape[1],
                                                  hidden_dim1=32,
                                                  hidden_dim2=64,
                                                  dropout=0.1,
                                                  device='cpu')

    adverarial_model.fit(torch.tensor(train_x.values, dtype=torch.float32),
                         torch.tensor(train_y.values, dtype=torch.float32), epochs=100)

    plt.plot(adverarial_model.learner_losses)
    plt.show()

    train_accuracy = (adverarial_model.predict(torch.tensor(train_x.values, dtype=torch.float32)) ==
            torch.tensor(train_y.values, dtype=torch.int64)).float().mean()

    test_accuracy = (adverarial_model.predict(torch.tensor(test_x.values, dtype=torch.float32)) ==
            torch.tensor(test_y.values, dtype=torch.int64)).float().mean()


    print(f"train acc: {train_accuracy}")
    print(f"test acc: {test_accuracy}")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
