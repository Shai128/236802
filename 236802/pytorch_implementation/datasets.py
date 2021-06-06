
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
        self.possible_protected_features = ['sex', 'race']

        self.target_feature_name = 'is_recid'
        self.folder_name = 'compas'
        self.dataset_name = 'Compas'

class Adult(DataSet):

    def __init__(self, ):
        super(DataSet, self).__init__()
        self.features = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income'
    ]
        self.features_to_transform = ["education", "marital-status", "occupation", "sex", "race",
        "relationship", "native-country", "income", "workclass"
    ]
        self.possible_protected_features = ['sex', 'race']

        self.target_feature_name = 'income'
        self.folder_name = 'adult'
        self.dataset_name = 'Adult'
