
from models import *
import pandas as pd
import matplotlib.pyplot as plt
from datasets import DataSet, Compas
from fairness_metrics import calculate_fairness_metrics
from torchvision.transforms import transforms
from utils import set_seeds
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

results_dir = 'results'

Models = [BaseModel, AdversarialReweightedModel, ImprovedModel]


if __name__ == '__main__':
    Models = [ImprovedModel]

    for Model in Models:
        for seed in tqdm(range(10)):
            set_seeds(seed)

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

            idx = np.random.permutation(len(df))
            train = df.iloc[idx[:len(train)]]

            test = df.iloc[idx[len(train):]]

            # protected_feature = dataset.possible_protected_features[0]  # 'sex', 'race',
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

            train_x, test_x, val_x = train.drop([target_name], axis=1), test.drop([target_name],axis=1), val.drop([target_name], axis=1)

            scaler = StandardScaler()
            scaler.fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)
            val_x = scaler.transform(val_x)

            model = Model(in_features=train_x.shape[1],
                                                          hidden_dim1=32,
                                                          hidden_dim2=64,
                                                          dropout=0.1,
                                                          lr=1e-4,
                                                          device='cpu')
            train_tensor_x = torch.tensor(train_x, dtype=torch.float32)
            train_tensor_y = torch.tensor(train_y.values, dtype=torch.float32)
            val_tensor_x = torch.tensor(val_x, dtype=torch.float32)
            val_tensor_y = torch.tensor(val_y.values, dtype=torch.long)
            model.fit(train_tensor_x,train_tensor_y, val_tensor_x, val_tensor_y, epochs=500)

            # model.plot_loss()

            train_preds = model.predict(torch.tensor(train_x, dtype=torch.float32))
            train_y = torch.tensor(train_y.values, dtype=torch.int64)
            train_accuracy = (train_preds == train_y).float().mean()

            test_preds = model.predict(torch.tensor(test_x, dtype=torch.float32))
            test_accuracy = (test_preds ==torch.tensor(test_y.values, dtype=torch.int64)).float().mean()
            y_scores = model.get_scores(torch.tensor(test_x, dtype=torch.float32)).detach().numpy()

            # print(f"train acc: {train_accuracy}")
            # print(f"test acc: {test_accuracy}")

            result_dict = calculate_fairness_metrics(test_private_features_values, protected_features, test_y, y_scores)
            method_name = Model.model_name
            results_name = f"{dataset.dataset_name}_{method_name}_seed={seed}"
            pd.DataFrame(result_dict, index=[results_name]).to_csv(f"{results_dir}/{results_name}.csv")
        # for protected_feature_value in range(max(test_private_feature_values)+1):
        #     idx = test_private_feature_values==protected_feature_value
        #     print(f"test protected feature value= {protected_feature_value} accuracy: {(test_preds[idx]== test_y[idx]).float().mean()}")



