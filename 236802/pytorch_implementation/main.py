
from models import *
import pandas as pd
from datasets import Compas, Adult
from fairness_metrics import calculate_fairness_metrics
from utils import set_seeds, load_data, create_folder_if_it_doesnt_exist
from tqdm import tqdm



def results_dir(dataset_name, method_name, seed):
    return f"results/{dataset_name}/{method_name}"

def result_path(dataset_name, method_name, seed):
    return f"{results_dir(dataset_name, method_name, seed)}/seed={seed}.csv"

Models = [BaseModel, AdversarialReweightedModel, ImprovedModel]
Models = [ImprovedModel]
Models = [BaseModel, AdversarialReweightedModel]


if __name__ == '__main__':
    # for switch in [1, 5, 10]:
    #     for lr in [1e-2, 1e-3, 1e-4]:
    #         for corr_mul in [0.01 , 0.05, 0.1, 0.5]:
    #             for adv_mul in [0.01 , 0.05, 0.1]:

                    for Model in Models:
                        for dataset in [Adult(), Compas()]:
                            # print(f"current model: {Model.model_name}")
                            for seed in tqdm(range(0, 20)):
                                set_seeds(seed)
                                # dataset = Adult()

                                train_x, test_x, val_x, train_y, test_y, val_y, \
                                 train_private_features_values, test_private_features_values, protected_features = load_data(dataset)

                                model = Model(in_features=train_x.shape[1],
                                              hidden_dim1=32,
                                              hidden_dim2=64,
                                              dropout=0.1,
                                              # lr=lr,
                                              device='cpu',
                                              take_best_model=False,
                                              # switch_count=switch,
                                              # discriminator_mult=adv_mul,
                                              # decorr_mult=corr_mul
                                          )

                                train_tensor_x = torch.tensor(train_x, dtype=torch.float32)
                                train_tensor_y = torch.tensor(train_y.values, dtype=torch.float32)
                                val_tensor_x = torch.tensor(val_x, dtype=torch.float32)
                                val_tensor_y = torch.tensor(val_y.values, dtype=torch.long)
                                model.fit(train_tensor_x,train_tensor_y, val_tensor_x, val_tensor_y, epochs=500, batch_size=512)

                                # model.plot_loss()

                                train_preds = model.predict(torch.tensor(train_x, dtype=torch.float32))
                                train_y = torch.tensor(train_y.values, dtype=torch.int64)
                                train_accuracy = (train_preds == train_y).float().mean()

                                test_preds = model.predict(torch.tensor(test_x, dtype=torch.float32))
                                test_accuracy = (test_preds ==torch.tensor(test_y.values, dtype=torch.int64)).float().mean()
                                y_scores = model.get_scores(torch.tensor(test_x, dtype=torch.float32)).detach().numpy()

                                result_dict = calculate_fairness_metrics(test_private_features_values, protected_features, test_x, test_y, y_scores)
                                result_dict['Train Accuracy'] = train_accuracy.item()
                                # method_name = f"{Model.model_name}_lr:{lr}_corr:{corr_mul}_adv_mul:{adv_mul}_switch:{switch}"
                                method_name = Model.model_name
                                path = result_path(dataset.dataset_name, method_name, seed)
                                create_folder_if_it_doesnt_exist(results_dir(dataset.dataset_name, method_name, seed))
                                pd.DataFrame(result_dict, index=[seed]).to_csv(path)
