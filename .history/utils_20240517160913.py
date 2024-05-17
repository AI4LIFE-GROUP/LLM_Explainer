# import contextlib
# import joblib
import numpy as np
import os
import json
import pickle
import torch.nn as nn
import re
import pandas as pd
from openxai.ML_Models.LR.model import LogisticRegression
import openxai.ML_Models.ANN.MLP as model_MLP
import datetime


# Make a numpy array of ints from 1 to N where each row is shuffled without replacement
def shuffled_indices(num_samples, num_feats):
    return np.array([np.random.choice(num_feats, num_feats, replace=False) for _ in range(num_samples)])


def DefineModel(num_feats, model_name, dim_per_layer=None, activation_per_layer=None):
    if 'ann' in model_name:
        dim_per_layer = [num_feats] + dim_per_layer
        model         = model_MLP.MLP(dim_per_layer, activation_per_layer)
    elif model_name == 'lr':
        dim_per_layer = [num_feats] + dim_per_layer
        model         = LogisticRegression(dim_per_layer[0], dim_per_layer[1])

    return model


def get_model_architecture(model_name):
    dim_per_layer_per_MLP = {'ann_s': [16, 2],
                             'ann_m': [32, 16, 2],
                             'ann_l': [64, 32, 16, 2],
                             'ann_xl': [256, 128, 64, 32, 16, 2],
                             'lr': [2]
                             }  # dimension for each layer for each network to train, ignoring input layer size
    activation_per_layer_per_MLP = {'ann_s': [nn.ReLU(), None],
                                    'ann_m': [nn.ReLU(), nn.ReLU(), None],
                                    'ann_l': [nn.ReLU(), nn.ReLU(), nn.ReLU(), None],
                                    'ann_xl': [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), None],
                                    'lr':    [None]
                                    } # ignore input layer size
    return dim_per_layer_per_MLP[model_name], activation_per_layer_per_MLP[model_name]

def get_model_names(model_name, dataset_name, base_model_dir):
    model_names = ['']['lr', 'ann_s', 'ann_m', 'ann_l', 'ann_xl']
    model_dirs  = [base_model_dir + model_name.upper() + '/' for model_name in model_names]

    model_dirs = dict(zip(model_names, model_dirs))

    if 'ClassWeighted_scale_minmax' in base_model_dir:
        compas_model_names = [
            '20230629_0056_2__compas_lr_0.001_auc_roc_0.82.pt',
            '20230629_0056_16_2__compas_ann_s_0.001_auc_roc_0.83.pt',
            '20230629_0056_32_16_2__compas_ann_m_0.001_auc_roc_0.83.pt',
            '20230629_0057_64_32_16_2__compas_ann_l_0.001_auc_roc_0.83.pt',
            '20230629_0057_256_128_64_32_16_2__compas_ann_xl_0.001_auc_roc_0.82.pt']

        credit_model_names = [
            '20230629_0101_2__credit_lr_0.001_auc_roc_0.81.pt',
            '20230629_0105_16_2__credit_ann_s_0.001_auc_roc_0.81.pt',
            '20230629_0109_32_16_2__credit_ann_m_0.001_auc_roc_0.81.pt',
            '20230629_0111_64_32_16_2__credit_ann_l_0.001_auc_roc_0.81.pt',
            '20230629_0113_256_128_64_32_16_2__credit_ann_xl_0.001_auc_roc_0.81.pt']

        adult_model_names = [
            '20230629_0039_2__adult_lr_0.001_auc_roc_0.89.pt',
            '20230629_0041_16_2__adult_ann_s_0.001_auc_roc_0.90.pt',
            '20230629_0044_32_16_2__adult_ann_m_0.001_auc_roc_0.90.pt',
            '20230629_0048_64_32_16_2__adult_ann_l_0.001_auc_roc_0.90.pt',
            '20230629_0051_256_128_64_32_16_2__adult_ann_xl_0.001_auc_roc_0.90.pt']

        blood_model_names = [
            '20230907_1208_2__blood_lr_0.001_auc_roc_0.66.pt',
            '20230907_1208_16_2__blood_ann_s_0.001_auc_roc_0.73.pt',
            '20230907_1208_32_16_2__blood_ann_m_0.001_auc_roc_0.74.pt',
            '20230907_1208_64_32_16_2__blood_ann_l_0.001_auc_roc_0.75.pt',
            '20230907_1208_256_128_64_32_16_2__blood_ann_xl_0.001_auc_roc_0.76.pt']

        model_file_names_data = {
            'compas': dict(zip(model_names, compas_model_names)),
            'adult': dict(zip(model_names, adult_model_names)),
            'credit': dict(zip(model_names, credit_model_names)),
            'blood': dict(zip(model_names, blood_model_names))
        }
    elif 'ClassWeighted_scale_standard' in base_model_dir:
        compas_model_names = ['20230713_1728_2__compas_lr_0.001_auc_roc_0.84.pt']
        credit_model_names = ['20230713_1730_2__credit_lr_0.001_auc_roc_0.81.pt']
        adult_model_names  = ['20230713_1728_2__adult_lr_0.001_auc_roc_0.90.pt']
        beauty_model_names = ['20240326_1629_2__beauty_lr_0.001_auc_roc_0.93.pt']

        model_file_names_data = {
            'compas': dict(zip(model_names, compas_model_names)),
            'adult': dict(zip(model_names, adult_model_names)),
            'credit': dict(zip(model_names, credit_model_names))
        }
    elif 'ClassWeighted' in base_model_dir:
        beauty_model_names = ['20240326_1629_2__beauty_lr_0.001_auc_roc_0.93.pt',
                              'none_ann_s',
                              'none_ann_m',
                              '20240328_1159_64_32_16_2__beauty_ann_l_0.001_auc_roc_0.94.pt',
                              'none_ann_xl']
        compas_model_names = ['']
        model_file_names_data = {
            'beauty': dict(zip(model_names, beauty_model_names))
        }
    else:
        raise NotImplementedError(f'Not implemented for {base_model_dir}')

    model_dir       = model_dirs[model_name]
    model_file_name = model_file_names_data[dataset_name][model_name]

    return model_dir, model_file_name


def append_k(ks, metrics):
    #metrics is a list of strings denoting the metrics
    #ks is a list of ints denoting the k values to append to each metric
    # append -1, -3, and -5 (for each k in ks) to each metric name
    appended_metrics = []
    for k in ks:
        temp = []
        for metric in metrics:
            temp.append(metric + '_' + str(k))
        appended_metrics.append(temp)
    #flatten the list
    metrics = [item for sublist in appended_metrics for item in sublist]
    return metrics


def _saveObjAsPkl(output_dir, file_name, data_to_save, extra_str=''):
    # Save to .pkl
    fpth = os.path.join(output_dir, file_name + extra_str + '.pkl')
    file = open(fpth, "wb")
    pickle.dump(data_to_save, file)
    file.close()

# recursively take a dict and write it out to a file with proper indentation and newlines for readability
# but if it's long make sure it doesnt write out "..." for the middle of the dict
def _writeDictToFile(f, d, indent=0):
    # f.write('\n')
    # for key, value in d.items():
    #     f.write('\t' * indent + str(key) + ':')
    #     if isinstance(value, dict):
    #         _writeDictToFile(f, value, indent+1)
    #         f.write('\n')
    #     else:
    #         f.write('\t' + str(value) + '\n')
    f.write('\n')
    for key, value in d.items():
        f.write('\t' * indent + str(key) + ':')
        if isinstance(value, dict):
            _writeDictToFile(f, value, indent+1)
            f.write('\n')
        elif isinstance(value, pd.DataFrame):
            f.write('\n' + value.to_string() + '\n')
        else:
            f.write('\t' + str(value) + '\n')

def saveParameters(output_dir, file_name, params, extra_str=''):
    # Save parameter dict items to a .txt file and .pkl
    fpth = os.path.join(output_dir, file_name+extra_str+'.txt')

    paramTxt = open(fpth, 'w')
    _writeDictToFile(paramTxt, params)
    paramTxt.close()

    # Save to .pkl
    _saveObjAsPkl(output_dir, file_name, params, extra_str)
    return


def _load_config(config_file):
    """Load config from file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def loadOpenAPIKeyFromFile(file_name):
    # Load OpenAI API key from file and remove newline character
    with open(file_name, 'r') as f:
        API_KEY = f.readline().replace('\n', '')
    return API_KEY

def mergeListOfStringIntoSingleString(LLM_topks):
    # merge list of strings into a single string. (for post-processing analysis)
    concated_LLM_topks = []
    for l, LLM_topk in enumerate(LLM_topks):
        concated_str = ''
        for k, feat in enumerate(LLM_topk):
            concated_str += feat
        concated_LLM_topks.append(concated_str)
    return concated_LLM_topks

def getExperimentID():
    date_info = datetime.datetime.now()
    testID = '%d%02d%02d_%02d%02d%02d' % (
    date_info.year, date_info.month, date_info.day, date_info.hour, date_info.minute, date_info.second)
    return testID

def getTotalCosts(folders):
    total_costs = []
    for folder in folders:
        total_costs.append(pd.read_csv(folder + 'total_costs.csv')['total_cost'].values)
    return np.array(total_costs)[:, 0]

def SaveExperimentInfo(config, folder_name_exp_id, n_shot, LLM, model_name, data_name, LLM_topks,
                       eval_min_idx, eval_max_idx, hidden_ys=None):
    # (Pickle) Save all replies of top-k features for each test instance individually
    output_dir       = config['output_dir'] + folder_name_exp_id + '/'
    output_file_name = 'n_shot_' + str(n_shot) + '_' + data_name + '_' + model_name + '_LLM_topK.pkl'
    with open(output_dir + output_file_name, 'wb') as file:
        pickle.dump(LLM_topks, file)

    if hidden_ys is not None:
        with open(output_dir + 'hidden_ys.pkl', 'wb') as file:
            pickle.dump(hidden_ys, file)

    # Save config file as json to output directory
    saveParameters(output_dir, 'pipeline_config', config)
    with open(f'{output_dir}/pipeline_config.json', 'w') as f:
        json.dump(config, f)

    # Write out all the final replies for each test sample to a single .txt file
    concated_replies = mergeListOfStringIntoSingleString(LLM_topks)
    file_name        = LLM + '_' + model_name.upper() + '_' + data_name + '_Replies.txt'
    paramTxt         = open(output_dir + file_name, 'w')
    for i in range(eval_min_idx, eval_max_idx):
        paramTxt.write('Sample ' + str(i) + ':\t' + str(list(concated_replies[i])) + '\n')
    paramTxt.close()

def get_k_words():
    return ['', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
           'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
           'twenty', 'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four', 'twenty-five', 'twenty-six',
           'twenty-seven', 'twenty-eight', 'twenty-nine', 'thirty', 'thirty-one', 'thirty-two', 'thirty-three',
           'thirty-four', 'thirty-five', 'thirty-six', 'thirty-seven', 'thirty-eight', 'thirty-nine', 'forty',
           'forty-one', 'forty-two', 'forty-three', 'forty-four', 'forty-five', 'forty-six', 'forty-seven',
           'forty-eight', 'forty-nine', 'fifty', 'fifty-one', 'fifty-two', 'fifty-three', 'fifty-four',
           'fifty-five', 'fifty-six', 'fifty-seven', 'fifty-eight', 'fifty-nine', 'sixty', 'sixty-one',
           'sixty-two', 'sixty-three', 'sixty-four', 'sixty-five', 'sixty-six', 'sixty-seven', 'sixty-eight',
           'sixty-nine', 'seventy', 'seventy-one', 'seventy-two', 'seventy-three', 'seventy-four', 'seventy-five',
           'seventy-six', 'seventy-seven', 'seventy-eight', 'seventy-nine', 'eighty', 'eighty-one', 'eighty-two',
           'eighty-three', 'eighty-four', 'eighty-five', 'eighty-six', 'eighty-seven', 'eighty-eight',
           'eighty-nine', 'ninety', 'ninety-one', 'ninety-two', 'ninety-three', 'ninety-four', 'ninety-five',
           'ninety-six', 'ninety-seven', 'ninety-eight', 'ninety-nine', 'one hundred']

def convert_to_int(samples_text):
    # Find all decimal numbers in the text
    decimals = re.findall(r'(\d+\.\d+)', samples_text)
    
    # Find the maximum number of decimal places
    max_decimal_places = max([len(dec.split('.')[1]) for dec in decimals])
    
    # Calculate multiplier factor
    factor = 10 ** max_decimal_places
    
    for dec in decimals:
        # Replace the decimal number with its integer equivalent in the text
        samples_text = samples_text.replace(dec, str(int(float(dec) * factor)), 1) # Replace only the first occurrence
    
    return samples_text
