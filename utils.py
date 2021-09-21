import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def load_dataset(dataset_path, dataset_id):
    if dataset_id == "adult":
        """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult
        and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617
        """
        raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                                 delimiter=', ', dtype=str)

        #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational_num', 'marital_status', 'occupation',
                        'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
                        'native_country', 'income']

        adult_data = pd.DataFrame(raw_data, columns=column_names)

        adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
        adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
        adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
        adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

        adult_data = adult_data.replace({'occupation': {'Adm-clerical':'White-Collar', 'Craft-repair':'Blue-Collar',
                                               'Exec-managerial':'White-Collar', 'Farming-fishing':'Blue-Collar',
                                                'Handlers-cleaners': 'Blue-Collar', 'Machine-op-inspct':'Blue-Collar',
                                                'Other-service':'Service', 'Priv-house-serv':'Service',
                                                'Prof-specialty':'Professional', 'Protective-serv':'Service',
                                                'Tech-support':'Service', 'Transport-moving':'Blue-Collar',
                                                'Unknown':'Other/Unknown', 'Armed-Forces':'Other/Unknown',
                                                '?':'Other/Unknown'}})

        adult_data = adult_data.replace({'marital_status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                            'Married-spouse-absent':'Married', 'Never-married':'Single'}})

        adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

        result = adult_data[['age', 'workclass', 'educational_num', 'marital_status', 'occupation',
                                 'race', 'gender', 'hours_per_week', 'income']]

        # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
        result = result.astype({"age": np.int64, "hours_per_week": np.int64, "educational_num": np.int64})
    elif dataset_id == "german":
        """Load the dataset from local hard drive and prepare it for data analysis"""
        german_data = pd.read_csv(dataset_path + '/german/German_credit_final.csv', header=None)

        column_names = ['duration', 'credit_history', 'credit_amount', 'employment', 'sex', 'age', 'housing', 'job',
                        'number_of_dependents', 'credit_risk']

        german_data.columns = column_names
        result = german_data.astype({"duration": np.int64, "credit_history": object, "credit_amount": np.float32,
                                "employment": object, "sex": object, "age": np.float32, "housing": object, "job": object,
                                "number_of_dependents": np.int64})
    elif dataset_id == "sangiovese":
        dataset = pd.read_csv(dataset_path + "\sangiovese\sangiovese.csv", index_col=None)
        dataset = dataset.drop(columns=['Unnamed: 0'])
        outcome = []
        for i in range(dataset.shape[0]):
            if dataset['GrapeW'][i] > 0:
                outcome.append(1)
            else:
                outcome.append(0)
        dataset['outcome'] = pd.Series(outcome)
        dataset.drop(columns=['GrapeW'],axis=1,inplace=True)

        result = dataset.astype({"SproutN": np.float32, "BunchN": np.float32, "WoodW": np.float32, "SPAD06": np.float32,
                                 "NDVI06": np.float32, "SPAD08": np.float32, "NDVI08": np.float32, "Acid": np.float32,
                                 "Potass": np.float32, "Brix": np.float32, "pH": np.float32, "Anthoc": np.float32,
                                 "Polyph": np.float32})

    return result

class proc_data:
    def __init__(self, args):
        self.data_df = args['dataframe']
        self.tst_size = args['test_size']
        self.seed = args['seed']

        self.outcome_name = self.data_df.iloc[:, -1].name
        self.feature_names = list(self.data_df.iloc[:, 0:-1].columns)

        # separate categorical and non categorical data
        cat_data = self.data_df.iloc[:,0:-1].select_dtypes(include=['object']).copy()
        self.categorical_feature_names = list(cat_data.columns)
        self.categorical_feature_indexes = [i for i in range(self.data_df.shape[1] - 1)
                                            if self.feature_names[i] in self.categorical_feature_names]

        cont_data = self.data_df.iloc[:,0:-1].select_dtypes(include=['number']).copy()
        self.continuous_feature_names = list(cont_data.columns)
        self.continuous_feature_indexes = [i for i in range(self.data_df.shape[1] - 1)
                                            if self.feature_names[i] in self.continuous_feature_names]

        if len(self.categorical_feature_names) > 0:
            self.data_df[self.categorical_feature_names] = self.data_df[self.categorical_feature_names].astype(
                'category')
            self.one_hot_encoded_data = self.one_hot_encoder(self.data_df)
            self.encoded_feature_names = [x for x in self.one_hot_encoded_data.columns.tolist(
            ) if x not in np.array([self.outcome_name])]
            for feature_name in self.continuous_feature_names:
                max_value = self.one_hot_encoded_data[feature_name].max()
                min_value = self.one_hot_encoded_data[feature_name].min()
                self.one_hot_encoded_data[feature_name] = (self.one_hot_encoded_data[feature_name] - min_value) / (max_value - min_value)
        else:
            self.one_hot_encoded_data = self.data_df.iloc[:,0:-1]
            for feature_name in self.continuous_feature_names:
                max_value = self.one_hot_encoded_data[feature_name].max()
                min_value = self.one_hot_encoded_data[feature_name].min()
                self.one_hot_encoded_data[feature_name] = (self.one_hot_encoded_data[feature_name] - min_value) / (max_value - min_value)
            self.encoded_feature_names = self.feature_names

        if len(self.continuous_feature_names) > 0:
            for feature in self.continuous_feature_names:
                if ((self.data_df[feature].dtype == np.float64) or (self.data_df[feature].dtype == np.float32)):
                    self.data_df[feature] = self.data_df[feature].astype(np.float32)
                else:
                    self.data_df[feature] = self.data_df[feature].astype(np.int64)

        self.train_df, self.test_df = self.split_data(one_hot=False)
        self.train_df_one_hot_encoded, self.test_df_one_hot_encoded = self.split_data()

        self.continuous_feature_range = self.get_features_range()

        self.encoded_categorical_feature_indexes = self.get_encoded_categorical_feature_indexes()
        self.minx, self.maxx = self.get_min_max()
        flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]


    def get_features_range(self):
        '''Returns the ranges of continuous features'''
        ranges = {}
        for feature_name in self.continuous_feature_names:
            ranges[feature_name] = [self.train_df[feature_name].min(), self.train_df[feature_name].max()]
        return ranges

    def one_hot_encoder(self, data):
        """One-hot-encodes the data."""
        return pd.concat([pd.get_dummies(data[col], prefix=col) if col in self.categorical_feature_names
                          else data[col] for col in self.feature_names], axis=1)

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]"""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.continuous_feature_range[feature_name][1]
            min_value = self.continuous_feature_range[feature_name][0]
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    def get_min_max(self):
        """Returns the min/max value of features"""
        minx = np.array([[0.0]*len(self.encoded_feature_names)])
        maxx = np.array([[1.0]*len(self.encoded_feature_names)])

        list_encoded_categorical_feature_indexes = [ind for sublist in self.encoded_categorical_feature_indexes for ind
                                                    in sublist]
        for idx, feature_name in enumerate(self.encoded_feature_names):
            if idx not in list_encoded_categorical_feature_indexes:
                minx[0][idx] = self.continuous_feature_range[feature_name][0]
                maxx[0][idx] = self.continuous_feature_range[feature_name][1]

        return minx, maxx

    def get_encoded_categorical_feature_indexes(self):
        """Returns the indexes of categorical features after one-hot-encoding."""
        cols = []
        for col_parent in self.categorical_feature_names:
            temp = [self.encoded_feature_names.index(
                col) for col in self.encoded_feature_names if col.startswith(col_parent)]
            cols.append(temp)
        return cols

    def prepare_df_for_encoding(self):
        """Generates an empty dataframe with same columns as encoded features"""
        levels = []
        colnames = self.categorical_feature_names
        for cat_feature in colnames:
            levels.append(self.data_df[cat_feature].cat.categories.tolist())

        if len(colnames) > 0:
            df = pd.DataFrame({colnames[0]: levels[0]})
        else:
            df = pd.DataFrame()

        for col in range(1, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: levels[col]})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        colnames = self.continuous_feature_names
        for col in range(0, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: []})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        return df

    def split_data(self, one_hot=True):
        ''' Splits data into train/test sets '''
        if one_hot:
            train_df, test_df = train_test_split( self.one_hot_encoded_data,
                test_size=int(np.ceil(self.tst_size * self.one_hot_encoded_data.shape[0])), random_state=self.seed)
        else:
            train_df, test_df = train_test_split( self.data_df,
                test_size=int(np.ceil(self.tst_size * self.data_df.shape[0])), random_state=self.seed)
        return train_df, test_df

class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

def proximity_plot(dataset_id, dataset_path, CFs_path):
    params = {'legend.fontsize': 'large',
              'figure.figsize': (5, 5),
              'axes.labelsize': 'large',
              'axes.titlesize':'large',
              'xtick.labelsize':'large',
              'ytick.labelsize':'large'}
    
    pylab.rcParams.update(params)

    df = load_dataset(dataset_path, dataset_id)
    data_params = {'dataframe': df, 'test_size': 0.2, 'seed': 17}
    data = proc_data(data_params)

    original_queries = pd.read_csv('all_queries_dataset_' + dataset_id + '.csv', header=0)
    distances_cat = np.zeros([5, 2, 9])
    distances_cont = np.zeros([5, 2, 9])
    cfs_ind = 0
    for total_cfs in [2, 4, 6, 8, 10]:
        for causal_flag in [True, False]:
            for query_instance in range(1, 10):
                f = pd.read_csv(CFs_path + 'cfs_total_'+str(total_cfs) + '_query_instance_'+str(query_instance) +
                                '_causal' + str(causal_flag) + '.csv', header=0)
                d_cat = []
                d_cont = []
                for i in range(f.shape[0]):
                    d_ij_cat = 0
                    d_ij_cont = 0

                    if len(data.categorical_feature_names) != 0:
                        for feature in data.categorical_feature_names:
                            d_ij_cat += int(f.loc[i][feature] != original_queries.loc[query_instance-1][feature])
                        d_ij_cat /= len(data.categorical_feature_names)
                        d_ij_cat = -d_ij_cat

                    for feature in data.continuous_feature_names:
                        if np.median(abs(data.train_df[feature] - np.median(data.train_df[feature]))) == 0:
                            d_ij_cont += abs(f.loc[i][feature] - original_queries.loc[query_instance-1][feature])
                        else:
                            d_ij_cont += abs(f.loc[i][feature] - original_queries.loc[query_instance-1][feature]) / np.median(
                                abs(data.train_df[feature] - np.median(data.train_df[feature])))
                    d_ij_cont /= len(data.continuous_feature_names)
                    d_ij_cont = -d_ij_cont

                    d_cat.append(d_ij_cat)
                    d_cont.append(d_ij_cont)
                if causal_flag == True:
                    distances_cat[cfs_ind, 0, query_instance-1] = np.mean(d_cat)
                    distances_cont[cfs_ind, 0, query_instance-1] = np.mean(d_cont)
                else:
                    distances_cat[cfs_ind, 1, query_instance-1] = np.mean(d_cat)
                    distances_cont[cfs_ind, 1, query_instance-1] = np.mean(d_cont)
        cfs_ind += 1

    mean_dists_cat = distances_cat.mean(axis=2)
    mean_dists_cont = distances_cont.mean(axis=2)
    print(mean_dists_cat)
    print(mean_dists_cont)

    legend_properties = {'weight':'bold'}
    if len(data.categorical_feature_names) != 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(mean_dists_cat[:, 0], color='blue', marker='o', label='C-DiCE')
        ax.plot(mean_dists_cat[:, 1], color='green', marker='s', label='DiCE')
        plt.ylim([-0.5, 0.5])
        plt.yticks(weight='bold')
        plt.xticks([0, 1, 2, 3, 4], [2, 4, 6, 8, 10], weight='bold')
        ax.set_xlabel('#CF', fontsize=20, labelpad=10)
        ax.set_ylabel('Categorical Proximity', fontsize=20, labelpad=10)
        plt.legend(loc="lower right", prop=legend_properties)
        plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_dists_cont[:, 0], color='blue', marker='o', label='C-DiCE')
    ax.plot(mean_dists_cont[:, 1], color='green', marker='s', label='DiCE')
    plt.ylim([np.floor(min(min(mean_dists_cont[:, 0]), min(mean_dists_cont[:, 1]))),
            np.ceil(max(max(mean_dists_cont[:, 0]), max(mean_dists_cont[:, 1])))])
    plt.yticks(weight='bold')
    plt.xticks([0, 1, 2, 3, 4], [2, 4, 6, 8, 10], weight='bold')
    ax.set_xlabel('#CF', fontsize=20, labelpad=10)
    ax.set_ylabel('Continuous Proximity', fontsize=20, labelpad=10)
    plt.legend(loc="lower right", prop=legend_properties)
    plt.show()