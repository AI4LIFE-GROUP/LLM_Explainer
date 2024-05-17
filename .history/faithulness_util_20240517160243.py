import torch
from openxai.evaluator import Evaluator
import os
from sklearn.metrics import auc
from utils import saveParameters
import numpy as np
import string
import pandas as pd


def generate_LLM_mask(num_features, top_k):
    # Assumes LLM_topk is sorted from most important to least important
    mask = torch.zeros(num_features, dtype=torch.bool)
    for i in range(top_k):
        mask[i] = True
    mask = ~mask
    return mask


def generate_mask(explanation, top_k):
    if not isinstance(explanation, torch.Tensor):
        explanation = torch.Tensor(explanation)
    mask_indices = torch.topk(explanation.abs(), top_k).indices
    mask = torch.ones(explanation.shape, dtype=bool)
    for i in mask_indices:
        mask[i] = False
    return mask

def makeFakeRankMagnitudesForFaithfulnessCalculation(LLM_topks, num_features):
    # This function makes fake ranked magnitudes for each test sample's top-k replies

    # Make dictionary with key alphabet A- Z in strings, and the value will be the index of the alphabet
    alphabet = string.ascii_uppercase
    alphabet_dict = {}
    for i, letter in enumerate(alphabet):
        alphabet_dict[letter] = i

    # Make an array of non-negative ints for each test sample's top-k rank. 0 == least important. k-1 == most important
    explanations = []  # fake ranked magnitudes
    for LLM_topk in LLM_topks:
        feature_importance = np.zeros(num_features)
        for i, letter in enumerate(LLM_topk):
            feature_importance[alphabet_dict[letter]] = num_features - i
        explanations.append(feature_importance)
    explanations = np.array(explanations)
    return explanations

def constructReplies(eval_min_idx, eval_max_idx, all_topks, orig_inds, unsolvable_idx):
    replies_df = pd.DataFrame(columns=['index', 'good/bad/unsolvable', 'reply'])
    replies_df['index'] = np.arange(eval_min_idx, eval_max_idx)
    replies_df['reply'] = all_topks
    replies_df['good/bad/unsolvable'] = 'unsolvable'
    replies_df.loc[orig_inds, 'good/bad/unsolvable'] = 'good'
    unsolvable_idxs = np.arange(eval_min_idx, eval_max_idx)[unsolvable_idx]
    bad_reply_idxs = list(set(list(range(eval_min_idx, eval_max_idx))) - set(orig_inds) - set(unsolvable_idxs))
    replies_df.loc[bad_reply_idxs, 'good/bad/unsolvable'] = 'bad'
    return replies_df, unsolvable_idxs, bad_reply_idxs

def getICLFromTextFiles(output_dir, model_name, data_name,
                        llm_name, n_feat, n_shot, experiment_section='3.1'):
    # assumes n_round = 3
    input_str, output_str = 'Input: ', 'Output: '
    files = [f for f in os.listdir(output_dir) if f.endswith('_summary.txt')]
    files = sorted(files, key=lambda x: int(x.split('_')[0]))
    eval_min_idx, eval_max_idx = int(files[0].split('_')[0]), int(files[-1].split('_')[0]) + 1
    if experiment_section == '3.2':
        n_shot -= 1
    y = np.zeros((eval_max_idx-eval_min_idx, n_shot), dtype=int)
    X = np.zeros((eval_max_idx-eval_min_idx, n_shot, n_feat), dtype=float)
    for i in range(eval_min_idx, eval_max_idx):
        filename = output_dir + str(i) + f'_{llm_name}_{model_name.upper()}_{data_name}_summary.txt'
        with open(filename, 'r') as f:
            file_text = f.read()
        ICL_text = file_text.split('PROMPT_TEXT:')[-1].split('REPLY:')[0]
        y[i] = np.array([int(y.split('\n')[0]) for y in ICL_text.split(output_str)[1:1+n_shot]])
        X[i] = [[float(x.strip().split('\n')[0][3:]) for j, x in enumerate(X.split(',')) if j<n_feat] for X in ICL_text.split(input_str)[1:1+n_shot]]
    return X, y

def saveFaithfulnessMetrics(output_dir, FA_AUC, RA_AUC, PGU_AUC, PGI_AUC, orig_inds,
                            replies_df, output_file_write_type='a', extra_str=''):
    fpth     = os.path.join(output_dir, 'FaithfulnessResults' + extra_str + '.txt')
    paramTxt = open(fpth, output_file_write_type)

    N_samps = len(PGI_AUC)
    paramTxt.write('Faithfulness Results' + '\n')
    paramTxt.write('--- MEAN +/- STD ERROR ---\n')
    # save FA_AUC RA PGU and PGI as comma separated values
    paramTxt.write('FA, RA, PGU, PGI\n')
    paramTxt.write(str(round(np.mean(FA_AUC), 3)) + '+/-' + str(round(np.std(FA_AUC)/np.sqrt(N_samps), 3)) + ',')
    paramTxt.write(str(round(np.mean(RA_AUC), 3)) + '+/-' + str(round(np.std(RA_AUC)/np.sqrt(N_samps), 3)) + ',')
    paramTxt.write(str(round(np.mean(PGU_AUC), 3)) + '+/-' + str(round(np.std(PGU_AUC)/np.sqrt(N_samps), 3)) + ',')
    paramTxt.write(str(round(np.mean(PGI_AUC), 3)) + '+/-' + str(round(np.std(PGI_AUC)/np.sqrt(N_samps), 3)) + '\n')
    paramTxt.close()

    # save reply_df to csv
    replies_df.to_csv(output_dir + 'replies_df.csv', index=False)

    faithfulness_metrics = dict(
                                zip(
                                    ['FA', 'RA', 'PGU', 'PGI', 'orig_inds'],
                                    [FA_AUC, RA_AUC, PGU_AUC, PGI_AUC, orig_inds]
                                )
                            )

    saveParameters(output_dir, 'faithfulness_metrics_all' + extra_str, faithfulness_metrics)

def getFaithfulnessMetricsString(model, FAs, RAs, PGUs, PGIs):
    print("LENGTHS", len(FAs), len(RAs), len(PGUs), len(PGIs))
    N_samps = len(PGUs)

    # save FAs RA PGU and PGI as comma separated values
    if hasattr(model, 'return_ground_truth_importance'):
        # MEAN +/- STD ERROR
        # FA RA PGU PGI
        metric_str = \
            str(round(np.mean(FAs), 3)) + '+/-' + str(round(np.std(FAs) / np.sqrt(N_samps), 3)) + ',' +\
            str(round(np.mean(RAs), 3)) + '+/-' + str(round(np.std(RAs) / np.sqrt(N_samps), 3)) + ',' +\
            str(round(np.mean(PGUs), 3)) + '+/-' + str(round(np.std(PGUs) / np.sqrt(N_samps), 3)) + ',' +\
            str(round(np.mean(PGIs), 3)) + '+/-' + str(round(np.std(PGIs) / np.sqrt(N_samps), 3))
    else:
        # PGU PGI
        metric_str = str(round(np.mean(PGUs), 3)) + '+/-' + str(round(np.std(PGUs) / np.sqrt(N_samps), 3)) + ',' +\
              str(round(np.mean(PGIs), 3)) + '+/-' + str(round(np.std(PGIs) / np.sqrt(N_samps), 3))
    return metric_str


def calculateFaithfulnessAUC(model, explanations, inputs, min_idx, max_idx, perturbation, perturb_num_samples,
                             feature_types, max_k):

    FA_AUC, RA_AUC, PGU_AUC, PGI_AUC = [], [], [], []
    for index in range(min_idx, max_idx):
        input_dict = dict()
        input_dict['x'] = inputs[index].reshape(-1)
        input_dict['explanation_x'] = explanations[index]
        input_dict['model'] = model
        input_dict['perturb_method'] = perturbation
        input_dict['perturb_num_samples'] = perturb_num_samples
        input_dict['feature_metadata'] = feature_types

        if max_k > 1:
            auc_x = np.arange(max_k) / (max_k - 1)
        FA, RA, PGU, PGI = [], [], [], []
        for top_k in range(1, max_k + 1):
            # topk and mask
            input_dict['top_k'] = top_k
            input_dict['mask'] = generate_mask(explanations[index], top_k)
            evaluator = Evaluator(input_dict)
            if hasattr(model, 'return_ground_truth_importance'):
                FA.append(evaluator.evaluate(metric='FA')[1])
                RA.append(evaluator.evaluate(metric='RA')[1])
            PGU.append(evaluator.evaluate(metric='PGU'))
            PGI.append(evaluator.evaluate(metric='PGI'))
        if hasattr(model, 'return_ground_truth_importance'):
            if max_k > 1:
                FA_AUC.append(auc(auc_x, FA))
                RA_AUC.append(auc(auc_x, RA))
            else:
                FA_AUC.append(FA)
                RA_AUC.append(RA)
        if max_k > 1:
            PGU_AUC.append(auc(auc_x, PGU))
            PGI_AUC.append(auc(auc_x, PGI))
        else:
            PGU_AUC.append(PGU)
            PGI_AUC.append(PGI)

    return FA_AUC, RA_AUC, PGU_AUC, PGI_AUC

def calculateFaithfulness_noAUC(model, explanations, inputs, min_idx, max_idx, perturbation, perturb_num_samples,
                                feature_types, top_k):

    FAs, RAs, PGUs, PGIs = [], [], [], []
    for index in range(min_idx, max_idx):
        input_dict = dict()
        input_dict['x']                    = inputs[index].reshape(-1)
        input_dict['explanation_x']        = explanations[index]
        input_dict['model']                = model
        input_dict['perturb_method']       = perturbation
        input_dict['perturb_num_samples']  = perturb_num_samples
        input_dict['feature_metadata']     = feature_types
        input_dict['top_k']                = top_k
        input_dict['mask']                 = generate_mask(input_dict['explanation_x'], top_k)

        evaluator = Evaluator(input_dict)
        FA, RA, PGU, PGI = [], [], [], []
        if hasattr(model, 'return_ground_truth_importance'):
            FA.append(evaluator.evaluate(metric='FA')[1])
            RA.append(evaluator.evaluate(metric='RA')[1])
        PGU.append(evaluator.evaluate(metric='PGU'))
        PGI.append(evaluator.evaluate(metric='PGI'))
        if hasattr(model, 'return_ground_truth_importance'):
            FAs.append(FA)
            RAs.append(RA)

        PGUs.append(PGU)
        PGIs.append(PGI)

    metrics_str = getFaithfulnessMetricsString(model, FAs, RAs, PGUs, PGIs)
    if hasattr(model, 'return_ground_truth_importance'):
        print('--- MEAN +/- STD ERROR ---')
        print('FA\tRA\tPGU\tPGI')
    else:
        print('--- MEAN +/- STD ERROR ---')
        print('PGU\tPGI')
    print(metrics_str)
    return FAs, RAs, PGUs, PGIs


def calculateFaithfulness(model, explanations, inputs, min_idx, max_idx, num_features, perturbation,
                          perturb_num_samples, feature_types, top_k, calculateAUC):
    if top_k == -1:
        top_k = num_features

    if not isinstance(explanations, torch.Tensor):
        explanations = torch.tensor(explanations)

    if calculateAUC:
        FAs, RAs, PGUs, PGIs = calculateFaithfulnessAUC(model, explanations, inputs, min_idx, max_idx, perturbation,
                                                        perturb_num_samples, feature_types, top_k)
        extra_str = '_AUC'
    else:
        FAs, RAs, PGUs, PGIs = calculateFaithfulness_noAUC(model, explanations, inputs, min_idx, max_idx, perturbation,
                                                           perturb_num_samples, feature_types, top_k)
        extra_str = ''

    metrics_str = getFaithfulnessMetricsString(model, FAs, RAs, PGUs, PGIs)
    if hasattr(model, 'return_ground_truth_importance'):
        print('--- MEAN +/- STD ERROR ---')
        print('FA' + extra_str + '\tRA' + extra_str + '\tPGU' + extra_str + '\tPGI' + extra_str)
    else:
        print('--- MEAN +/- STD ERROR ---')
        print('PGU' + extra_str + '\tPGI' + extra_str)
    print(metrics_str)

    return FAs, RAs, PGUs, PGIs
