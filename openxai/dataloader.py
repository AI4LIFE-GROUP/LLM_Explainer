import csv
import io
import os
import subprocess
import torch
import pandas as pd
from openxai import dgp_synthetic
from errno import EEXIST
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from urllib.request import urlretrieve
import pickle as pkl
# from xai_benchmark.dataset.Synthetic import dgp_synthetic

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter


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
    def __init__(self, path, filename, label, download=False, scale='minmax', gauss_params=None, file_url=None,
                 dtype=''):

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
#
# class TextDataLoader(data.Dataset):
#     def __init__(self, path, filename, label, premade_data_name='', dtype=''):
#
#         """
#         Load training dataset
#         :param path: string with path to training set
#         :param filename: string with name of file
#         :param label: string, column name for label
#         :param dtype: string, type of data to load
#
#         :return: tensor with training data
#         """
#         if not dtype in ['train', 'val', 'test']:
#             raise NotImplementedError('The current version of DataLoader class only provides the following datatypes: {train, val, test}')
#
#         self.path = path
#
#         if not os.path.isfile(path + filename):
#             raise RuntimeError("Dataset not found. You can use download=True to download it")
#
#         # load data from pkl
#         with open(path + premade_data_name + '.pkl', 'rb') as f:
#             self.input = pkl.load(f)
#
#         self.dataset = pd.read_pickle(path + filename)
#         self.target = label
#         self.targets = np.array(self.dataset[self.target])
#
#         self.dataset = np.array(self.dataset.drop(columns=[self.target]))
#
#         val_percent_of_train  = 0.2
#         n_train               = self.dataset.shape[0]
#         n_val                 = int(n_train * val_percent_of_train)
#
#         if dtype == 'train':
#             self.dataset = self.dataset[n_val:]
#             self.targets = self.targets[n_val:]
#             self.data    = self.input[n_val:]
#
#         elif dtype == 'val':
#             self.dataset = self.dataset[0:n_val]
#             self.targets = self.targets[0:n_val]
#             self.data    = self.input[0:n_val]
#         else:
#             self.data = self.input
#
#         self.sentences = self.dataset
#
#         self.tokenizer = get_tokenizer('basic_english')
#         self.word_counter = Counter()
#         for (line, label) in zip(self.dataset, self.targets):
#             self.word_counter.update(self.tokenizer(line[0]))
#         self.vocab = Vocab(self.word_counter, min_freq=10)
#
#         self.num_class = len(set(self.targets))
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#
#         # select correct row with idx
#         if isinstance(idx, torch.Tensor):
#             idx = idx.tolist()
#         batch = [(self.sentences[idx], self.targets[idx])]
#         labels = torch.tensor([label for _, label in batch])
#         text_list = [self.tokenizer(line[0]) for line, _ in batch]
#
#         # flatten tokens across the whole batch
#         text = torch.tensor([self.vocab[t] for tokens in text_list for t in tokens])
#         tokenized_list = [torch.tensor([self.vocab[t] for t in tokens]) for tokens in text_list]
#         # the offset of each example
#         offsets = torch.tensor(
#             [0] + [len(tokens) for tokens in text_list][:-1]
#         ).cumsum(dim=0)
#
#         return (labels, text, offsets, tokenized_list)
#
#         # return (self.data[idx], self.sentences[idx], self.targets.values[idx])
#
#     def get_vocab_size(self):
#         return len(self.vocab)
#
#     def get_vocab(self):
#         return self.vocab
#
#     def get_tokenizer(self):
#         return self.tokenizer
#
#     def get_number_of_features(self):
#         return self.data.shape[1]
#
#     def get_number_of_instances(self):
#         return self.data.shape[0]
#
#     def mkdir_p(self, mypath):
#         """Creates a directory. equivalent to using mkdir -p on the command line"""
#         try:
#             os.makedirs(mypath)
#         except OSError as exc:  # Python >2.5
#             if exc.errno == EEXIST and os.path.isdir(mypath):
#                 pass
#             else:
#                 raise


def get_text_dataset(dtype, path, filename, label):
    """
    Load training dataset
    :param path: string with path to training set
    :param filename: string with name of file
    :param label: string, column name for label
    :param dtype: string, type of data to load

    :return: tensor with training data
    """
    if not dtype in ['train', 'val', 'test']:
        raise NotImplementedError('The current version of DataLoader class only provides the following datatypes: {train, val, test}')

    if not os.path.isfile(path + filename):
        raise RuntimeError("Dataset not found. You can use download=True to download it")

    dataset = pd.read_pickle(path + filename)
    target = label
    targets = np.array(dataset[target])

    sentences = np.array(dataset.drop(columns=[target]))

    val_percent_of_train = 0.1
    n_train = dataset.shape[0]
    n_val = int(n_train * val_percent_of_train)

    if dtype == 'train':
        sentences = sentences[n_val:]
        targets = targets[n_val:]
    elif dtype == 'val':
        sentences = sentences[0:n_val]
        targets = targets[0:n_val]

    return sentences, targets

def get_tokenizer_and_vocab(X_train, y_train):
    tokenizer = get_tokenizer('basic_english')
    word_counter = Counter()
    for (line, label) in zip(X_train, y_train):
        word_counter.update(tokenizer(line[0]))
    voc = Vocab(word_counter, min_freq=1)

    print('Vocabulary size:', len(voc))
    return tokenizer, voc


# def collate_batch(batch, voc, tokenizer):
#     labels = torch.tensor([label for _, label in batch])
#     text_list = [tokenizer(line) for line, _ in batch]
#
#     # flatten tokens across the whole batch using indexing for the vocab
#     text = torch.tensor([voc[t] for tokens in text_list for t in tokens if t in voc.stoi])
#     tokenized_list = [torch.tensor([voc[t] for t in tokens if t in voc.stoi]) for tokens in text_list]
#     # the offset of each example
#     offsets = torch.tensor(
#         [0] + [len(tokens) for tokens in text_list][:-1]
#     ).cumsum(dim=0)
#
#     return labels, text, offsets, tokenized_list

# def collate_batch(batch, tokenizer, voc):
#     labels = torch.tensor([label for _, label in batch])
#     text_list = [tokenizer(line[0]) for line, _ in batch]
#
#     # flatten tokens across the whole batch
#     text = torch.tensor([voc[t] for tokens in text_list for t in tokens])
#     tokenized_list = [torch.tensor([voc[t] for t in tokens]) for tokens in text_list]
#     # the offset of each example
#     offsets = torch.tensor(
#         [0] + [len(tokens) for tokens in text_list][:-1]
#     ).cumsum(dim=0)
#
#     return labels, text, offsets, tokenized_list

def collate_batch(batch, tokenizer, voc):
    labels = torch.tensor([label for _, label in batch])
    text_list = [tokenizer(line[0]) for line, _ in batch]
    #pad the text_list to have the same length
    max_len = max([len(tokens) for tokens in text_list])
    for tokens in text_list:
        while len(tokens) < max_len:
            tokens.append('<pad>')

    inputs = torch.stack([torch.tensor([voc[t] for t in tokens]) for tokens in text_list])

    return labels, inputs


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
            'beauty': ('beauty', transform, 'sentiment'),
            'amazon_1000': ('amazon_1000', transform, 'sentiment'),
            'imdb': ('imdb', transform, 'sentiment'),
            'yelp': ('yelp', transform, 'sentiment'),
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
    elif dict[data_name][0] == 'beauty':
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = 'beauty-train.pkl'
        file_test = 'beauty-test.pkl'
    elif dict[data_name][0] == 'amazon_1000':
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = 'amazon_1000-train.pkl'
        file_test = 'amazon_1000-test.pkl'
    elif dict[data_name][0] == 'imdb':
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = 'imdb-train.pkl'
        file_test = 'imdb-test.pkl'
    elif dict[data_name][0] == 'yelp':
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = 'yelp-train.pkl'
        file_test = 'yelp-test.pkl'
    else:
        prefix = './data/' + dict[data_name][0] + '/'
        file_train = data_name + '-train.csv'
        file_test = data_name + '-test.csv'

    if data_name == 'beauty':
        # dataset_train = TextDataLoader(path=prefix, filename=file_train, label=dict[data_name][2], dtype='train', premade_data_name='beauty-train-embeddings')
        # dataset_val = TextDataLoader(path=prefix, filename=file_train, label=dict[data_name][2], dtype='val', premade_data_name='beauty-train-embeddings')
        # dataset_test = TextDataLoader(path=prefix, filename=file_test, label=dict[data_name][2], dtype='test', premade_data_name='beauty-test-embeddings')
        #
        # trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        # valloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        # testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        pass
    elif data_name == 'amazon_1000' or data_name == 'imdb' or data_name == 'yelp':
        X_train, y_train = get_text_dataset(dtype='train', path=prefix, filename=file_train, label=dict[data_name][2])
        X_val, y_val = get_text_dataset(dtype='val', path=prefix, filename=file_train, label=dict[data_name][2])
        X_test, y_test = get_text_dataset(dtype='test', path=prefix, filename=file_test, label=dict[data_name][2])

        tokenizer, voc = get_tokenizer_and_vocab(X_train, y_train)

        trainloader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda batch: collate_batch(batch, tokenizer, voc))
        valloader = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False,
                               collate_fn=lambda batch: collate_batch(batch, tokenizer, voc))
        testloader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False,
                                collate_fn=lambda batch: collate_batch(batch, tokenizer, voc))
    else:
        dataset_train = TabularDataLoader(path=prefix, filename=file_train, label=dict[data_name][2], scale=scaler,
                                          gauss_params=gauss_params, download=download,
                                          file_url=urls.get(file_train[:-4], None), dtype='train')

        dataset_val = TabularDataLoader(path=prefix, filename=file_train, label=dict[data_name][2], scale=scaler,
                                          gauss_params=gauss_params, download=download,
                                          file_url=urls.get(file_train[:-4], None), dtype='val')

        dataset_test = TabularDataLoader(path=prefix, filename=file_test, label=dict[data_name][2], scale=scaler,
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
        # do you want to treat all features as continuous or discrete?
        feature_types = ['c']*6  # ['c', 'c', 'c', 'd', 'd', 'd']
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
        # do you want to treat all features as continuous or discrete?
        feature_types = ['c'] * 13  # ['c', 'c', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
    elif dname == 'credit':
        num_features = 10
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
    # elif dname == 'heloc':
    #     num_features = 23
    #     feature_names = [''] * num_features
    #     suffixes      = [''] * num_features
    #     conversion    = [rounder] * num_features
    #     feature_types = ['c'] * num_features
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
