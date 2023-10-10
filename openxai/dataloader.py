import csv
import io
import os
import subprocess
import torch
import pandas as pd
from LLM_Explainer.openxai import dgp_synthetic
from errno import EEXIST
import torch.utils.data as data
from torch.utils.data import DataLoader
from urllib.request import urlretrieve
# from xai_benchmark.dataset.Synthetic import dgp_synthetic
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def download_file(url, filename):
    # Download the file from the URL
    subprocess.call(["wget", "-O", filename, url])

    with open(filename, "r") as f:
        data = f.read()

    # Detect the file format
    if '\t' in data:  # if the file is tab delimited
        # Convert the file to CSV format
        data = io.StringIO(data)
        reader = csv.reader(data, delimiter='\t')
        output = io.StringIO()
        writer = csv.writer(output)
        for row in reader:
            writer.writerow(row)
        data = output.getvalue()

        # Save the file to disk
        with open(filename, 'w', newline='') as f:
            f.write(data)


class TabularDataLoader(data.Dataset):
    def __init__(self, path, filename, label, download=False, scale='minmax', gauss_params=None, file_url=None, dtype=''):

        """
        Load training dataset
        :param path: string with path to training set
        :param label: string, column name for label
        :param scale: string; 'minmax', 'standard', or 'none'
        :param dict: standard params of gaussian dgp
        :return: tensor with training data
        """

        self.path = path

        # Load Synthetic dataset
        if 'Synthetic' in self.path:

            '''
            if download:
                url = 'https://raw.githubusercontent.com/chirag126/data/main/'
                self.mkdir_p(path)
                file_download = url + 'dgp_synthetic.py'
                # import ipdb; ipdb.set_trace()
                urlretrieve(file_download, path + 'dgp_synthetic.py')

            if not os.path.isdir(path + 'dgp_synthetic.py'):
                raise RuntimeError("Dataset not found. You can use download=True to download it")

            from openxai_ import dgp_synthetic

            '''

            if gauss_params is None:
                gauss_params = {
                    'n_samples': 2500,
                    'dim': 20,
                    'n_clusters': 10,
                    'distance_to_center': 5,
                    'test_size': 0.25,
                    'upper_weight': 1,
                    'lower_weight': -1,
                    'seed': 564,
                    'sigma': None,
                    'sparsity': 0.25
                }

            data_dict, data_dict_train, data_dict_test = dgp_synthetic.generate_gaussians(gauss_params['n_samples'],
                                                                                          gauss_params['dim'],
                                                                                          gauss_params['n_clusters'],
                                                                                          gauss_params['distance_to_center'],
                                                                                          gauss_params['test_size'],
                                                                                          gauss_params['upper_weight'],
                                                                                          gauss_params['lower_weight'],
                                                                                          gauss_params['seed'],
                                                                                          gauss_params['sigma'],
                                                                                          gauss_params['sparsity']).dgp_vars()

            self.ground_truth_dict = data_dict
            self.target = label

            if 'train' in filename:
                data_dict = data_dict_train
            elif 'test' in filename:
                data_dict = data_dict_test
            else:
                raise NotImplementedError('The current version of DataLoader class only provides training and testing splits')

            self.dataset = pd.DataFrame(data_dict['data'])
            data_y = pd.DataFrame(data_dict['target'])

            names = []
            for i in range(gauss_params['dim']):
                name = 'x' + str(i)
                names.append(name)

            self.dataset.columns = names
            self.dataset['y'] = data_y

            # add additional Gaussian related aspects
            self.probs = data_dict['probs']
            self.masks = data_dict['masks']
            self.weights = data_dict['weights']
            self.masked_weights = data_dict['masked_weights']
            self.cluster_idx = data_dict['cluster_idx']

        else:
            if download:
                self.mkdir_p(path)
                if file_url is None:
                    url = 'https://raw.githubusercontent.com/chirag126/data/main/'
                    file_download = url + filename
                    urlretrieve(file_download, path + filename)
                else:
                    download_file(file_url, path + filename)

            if not os.path.isfile(path + filename):
                raise RuntimeError("Dataset not found. You can use download=True to download it")
            self.dataset = pd.read_csv(path + filename)
            self.target = label
            self.targets = self.dataset[self.target]

            val_percent_of_train  = 0.2
            n_train               = self.dataset.shape[0]
            n_val                 = int(n_train * val_percent_of_train)
            if dtype == 'train':
                self.dataset = self.dataset[n_val:]
                self.targets = self.targets[n_val:]
            elif dtype == 'val':
                self.dataset = self.dataset[0:n_val]
                self.targets = self.targets[0:n_val]


        # Save target and predictors
        self.X = self.dataset.drop(self.target, axis=1)


        if not dtype in ['train', 'val', 'test']:
            raise NotImplementedError('The current version of DataLoader class only provides the following datatypes: {train, val, test}')

        # Save feature names
        self.feature_names = self.X.columns.to_list()
        self.target_name = label

        # Transform data
        if scale == 'minmax':
            self.scaler = MinMaxScaler()
        elif scale == 'standard':
            self.scaler = StandardScaler()
        elif scale == 'none':
            self.scaler = None
        else:
            raise NotImplementedError('The current version of DataLoader class only provides the following transformations: {minmax, standard, none}')

        if self.scaler is not None:
            self.scaler.fit_transform(self.X)
            self.data = self.scaler.transform(self.X)
        else:
            self.data = self.X.values


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # select correct row with idx
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        if 'Synthetic' in self.path:
            return (self.data[idx], self.targets.values[idx], self.weights[idx], self.masks[idx],
                    self.masked_weights[idx], self.probs[idx], self.cluster_idx[idx])
        else:
            return (self.data[idx], self.targets.values[idx])

    def get_number_of_features(self):
        return self.data.shape[1]

    def get_number_of_instances(self):
        return self.data.shape[0]

    def mkdir_p(self, mypath):
        """Creates a directory. equivalent to using mkdir -p on the command line"""
        try:
            os.makedirs(mypath)
        except OSError as exc:  # Python >2.5
            if exc.errno == EEXIST and os.path.isdir(mypath):
                pass
            else:
                raise


def return_loaders(data_name, download=False, batch_size=32, transform=None, scaler='minmax', gauss_params=None):

    # Create a dictionary with all available dataset names
    dict = {
            'adult': ('Adult', transform, 'income'),
            'compas': ('COMPAS', transform, 'risk'),
            'german': ('German_Credit_Data', transform, 'credit-risk'),
            'heloc': ('Heloc', transform, 'RiskPerformance'),
            'credit': ('Credit', transform, 'SeriousDlqin2yrs'),
            'synthetic': ('Synthetic', transform, 'y'),
            'rcdv': ('rcdv1980', transform, 'recid'),
            'lending-club': ('lending-club', transform, 'loan_repaid'),
            'student': ('student', transform, 'decision'),
            'blood': ('blood', transform, 'C'),
            }

    urls = {
            'rcdv-train': 'https://dataverse.harvard.edu/api/access/datafile/7093737',
            'rcdv-test': 'https://dataverse.harvard.edu/api/access/datafile/7093739',
            'lending-club-train': 'https://dataverse.harvard.edu/api/access/datafile/6767839',
            'lending-club-test': 'https://dataverse.harvard.edu/api/access/datafile/6767838',
            'student-train': 'https://dataverse.harvard.edu/api/access/datafile/7093733',
            'student-test': 'https://dataverse.harvard.edu/api/access/datafile/7093734',
            }

    if dict[data_name][0] == 'synthetic':
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = 'train'
        file_test = 'test'
    else:
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = data_name + '-train.csv'
        file_test = data_name + '-test.csv'

    dataset_train = TabularDataLoader(path=prefix, filename=file_train,
                                      label=dict[data_name][2], scale=scaler,
                                      gauss_params=gauss_params, download=download,
                                      file_url=urls.get(file_train[:-4], None), dtype='train')

    dataset_val = TabularDataLoader(path=prefix, filename=file_train,
                                      label=dict[data_name][2], scale=scaler,
                                      gauss_params=gauss_params, download=download,
                                      file_url=urls.get(file_train[:-4], None), dtype='val')

    dataset_test = TabularDataLoader(path=prefix, filename=file_test,
                                     label=dict[data_name][2], scale=scaler,
                                     gauss_params=gauss_params, download=download,
                                     file_url=urls.get(file_test[:-4], None), dtype='test')

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader



def get_feature_details(dname, n_round):
    # COMMENT FROM DL: Should we change Other to non-X?
    # e.g. (race is White, race is non-White) rather than (race is White, race is Other)
    rounder = lambda x: round(x, n_round)
    if dname=='compas':
        charge = lambda x: 'Misdemeanor' if x==0 else 'Felony'
        gender = lambda x: 'Male' if x==0 else 'Female'
        race = lambda x: 'Other' if x==0 else 'African-American'
        # Warning: editing feature names may exceed query token limit
        feature_names = ['Age', 'Number of Priors', 'Length of Stay',  # continuous
                         'Charge', 'Sex', 'Race']  # categorical
        conversion = [rounder, rounder, rounder, charge, gender, race]
        suffixes = [' Years', '', ' Months', '', '', '']
        feature_types = ['c']*6#['c', 'c', 'c', 'd', 'd', 'd']
    elif dname=='adult':
        gender = lambda x: 'Male' if x==1 else 'Female'
        workclass = lambda x: 'Private' if x==1 else 'Other'
        marital_status = lambda x: 'Non-Married' if x==1 else 'Married'
        # Comment from DL: occupation is either Other, or a specific occupation from the following list:
        # occupation: Tech-support, Craft-repair, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners,
        # Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        # so not sure if we should include this in the prompt later or when you show details or not
        occupation = lambda x: 'Other' if x==1 else 'Different'
        relationship = lambda x: 'Non-Husband' if x==1 else 'Husband'  # also not sure what non-husband means
        race = lambda x: 'White' if x==1 else 'Other'
        native_country = lambda x: 'US' if x==1 else 'Other'
        feature_names = ['Age', 'Final Weight', 'Education Number', 'Capital Gain', 'Capital Loss', 'Hours per Week',  # continuous
                         'Sex', 'Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Native Country']  # categorical
        conversion = [rounder, rounder, rounder, rounder, rounder, rounder,
                      gender, workclass, marital_status, occupation, relationship, race, native_country]
        # Comment from DL: Assuming capital gain and loss are in dollars
        suffixes = [' Years', '', '', ' Dollars', ' Dollars', ' Hours', '', '', '', '', '', '', '', '']
        feature_types = ['c'] * 13 # ['c', 'c', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
    elif dname == 'credit':
        #TODO - need to implement these if you want to show feature details
        num_features = 10
        feature_names = [''] * num_features
        suffixes      = [''] * num_features
        conversion    = [rounder] * num_features
        feature_types = ['c'] * num_features
    elif dname == 'heloc':
        # TODO - need to implement these if you want to show feature details
        num_features = 23
        feature_names = [''] * num_features
        suffixes      = [''] * num_features
        conversion    = [rounder] * num_features
        feature_types = ['c'] * num_features
    elif dname == 'blood':
        num_features = 4
        feature_names = [''] * num_features
        suffixes = [''] * num_features
        conversion = [rounder] * num_features
        feature_types = ['c'] * num_features
    # elif dname == 'german':
    # TODO - not sure how to represent these since we are using A-Z for the feature names in the prompt.
    #     num_features = 60
    #     feature_names = [''] * num_features
    #     suffixes      = [''] * num_features
    #     conversion    = [rounder] * num_features
    #     feature_types = ['c'] * 6 + ['d'] * 54
    # elif dname=='german': TO BE IMPLEMENTED
    #     feature_names = ['Duration', 'Amount', 'Installment Rate', 'Present Residence', 'Age', 'Number of Credits',
    #                      'People Liable', 'Foreign Worker', 'Status', 'Credit History', 'Purpose', 'Savings',
    #                      'Employment Duration', 'Personal Status Sex', 'Other Debtors', 'Property',
    #                      'Other Installment Plans', 'Housing', 'Job', 'Telephone']
    #     conversion = [rounder, rounder, rounder, rounder, rounder, rounder,
    #                   people_liable, foreign_worker, status,
    #                   credit_history, purpose, savings, employment_duration, personal_status_sex, other_debtors,
    #                   property, other_installment_plans, housing, job, telephone]
    #     suffixes = [' Months', ' Dollars', '', '', ' Years', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
    else:
        feature_types = None
        feature_names = None
        conversion    = None
        suffixes      = None
        print("CONVERSION NOT IMPLEMENTED FOR THIS DATASET")
        # raise NotImplementedError('Conversion not implemented for this dataset')
    return feature_types, feature_names, conversion, suffixes
