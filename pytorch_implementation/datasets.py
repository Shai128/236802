
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

