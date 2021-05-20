
from models import *
import pandas as pd
from datasets import Compas
from fairness_metrics import calculate_fairness_metrics
from utils import set_seeds, load_data
from tqdm import tqdm


results_dir = 'results'

Models = [BaseModel, AdversarialReweightedModel, ImprovedModel]


if __name__ == '__main__':
    Models = [ImprovedModel]
    for Model in Models:
        for seed in tqdm(range(0, 20)):

            set_seeds(seed)
            dataset = Compas()

            train_x, test_x, val_x, train_y, test_y, val_y, \
             train_private_features_values, test_private_features_values, protected_features = load_data(dataset)

            model = Model(in_features=train_x.shape[1],
                          hidden_dim1=32,
                          hidden_dim2=64,
                          dropout=0.1,
                          lr=1e-4,
                          device='cpu',
                          take_best_model=False
                          )

            train_tensor_x = torch.tensor(train_x, dtype=torch.float32)
            train_tensor_y = torch.tensor(train_y.values, dtype=torch.float32)
            val_tensor_x = torch.tensor(val_x, dtype=torch.float32)
            val_tensor_y = torch.tensor(val_y.values, dtype=torch.long)
            model.fit(train_tensor_x,train_tensor_y, val_tensor_x, val_tensor_y, epochs=500, batch_size=1024)

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



