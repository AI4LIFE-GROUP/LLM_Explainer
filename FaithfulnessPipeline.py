from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from llms.response import removeBadReplies
import copy
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from utils import _load_config, saveParameters
from openxai.LoadModel import DefineModel
from openxai.dataloader import return_loaders, get_feature_details
from faithulness_util import makeFakeRankMagnitudesForFaithfulnessCalculation, constructReplies,\
    calculateFaithfulness, saveFaithfulnessMetrics, getFaithfulnessMetricsString, getICLFromTextFiles
from llms.response import parseLLMTopKsFromTxtFiles, LoadLLMRepliesFromTextFiles
from utils import get_model_names, get_model_architecture


def runFaithfulnessPipeline(config = None):
    if config is None:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='faithfulness_config.json',
                            help='faithfulness .json file of parameters for calculating faithfulness metrics')

        # Get config dictionary
        args    = vars(parser.parse_args())
        config  = _load_config(args['config'])

    # Set config parameters
    perturbation_mean          = config['perturbation_mean']
    perturbation_std           = config['perturbation_std']
    perturb_num_samples        = config['perturb_num_samples']
    LLM_topks_file_name        = config['LLM_topks_file_name']
    eval_min_idx               = config['eval_min_idx']
    eval_max_idx               = config['eval_max_idx']
    SEED                       = config['SEED']
    data_scaler                = config['data_scaler']
    output_dir                 = config['output_dir']
    eval_top_k                 = config['eval_top_k']
    LLM_top_k                  = config['LLM_top_k']  # LLM_k is the number of top-k repllies asked for in the LLM
    save_results               = config['save_results']
    model_name                 = config['model_name']
    data_name                  = config['data_name']
    base_model_dir             = config['base_model_dir']
    load_reply_strategy        = config['load_reply_strategy']
    calculateAUC               = config['calculateAUC']
    experiment_section         = config['experiment_section']
    model_dir, model_file_name = get_model_names(model_name, data_name, base_model_dir)
    dim_per_layer_per_MLP, activation_per_layer_per_MLP = get_model_architecture(model_name)

    perturbation_flip_percentage = np.sqrt(2/np.pi)*perturbation_std

    np.random.seed(SEED)

    feature_types, feature_names, conversion, suffixes = get_feature_details(data_name, None)
    num_features = len(feature_types)
    perturbation = NormalPerturbation("tabular", mean=perturbation_mean,
                                      std_dev=perturbation_std, flip_percentage=perturbation_flip_percentage)

    # Load dataset
    download_data     = False if data_name in ['compas', 'blood'] else True
    _, _, loader_test = return_loaders(data_name=data_name, download=download_data, scaler=data_scaler)
    inputs            = torch.FloatTensor(loader_test.dataset.data)
    eval_max_idx      = min(1000, inputs.shape[0]) if eval_max_idx == -1 else eval_max_idx
    print("eval_min_idx : ", eval_min_idx)
    print("eval_max_idx : ", eval_max_idx)

    # Load model
    input_size = loader_test.dataset.get_number_of_features()
    model      = DefineModel(model_name, input_size, dim_per_layer_per_MLP,
                        activation_per_layer_per_MLP)
    model.load_state_dict(torch.load(model_dir + model_file_name))
    model.eval()

    n_shot = int(output_dir.split('nshot')[-1].split('_')[0])
    LLM_name = output_dir.rstrip('/').split('/')[-1].split('_')
    print('n_shot: ', n_shot)
    for i, name in enumerate(LLM_name):
        if not name.isnumeric():
            LLM_name = name
            break
    print('LLM_name: ', LLM_name)
    X_ICL, y_ICL = getICLFromTextFiles(output_dir, model_name, data_name,
                                       LLM_name, input_size, n_shot, experiment_section)
    # save X_ICL and y_ICL to numpy files
    np.save(output_dir + 'X_ICL.npy', X_ICL)
    np.save(output_dir + 'y_ICL.npy', y_ICL)
    unsolvable_idx = []
    for i in range(y_ICL.shape[0]):
        if len(np.unique(y_ICL[i])) == 1:
            unsolvable_idx.append(True)
        else:
            unsolvable_idx.append(False)
    unsolvable_idx = np.array(unsolvable_idx)
    orig_inds = np.arange(eval_min_idx, eval_max_idx)[~unsolvable_idx]

    if load_reply_strategy == 'pkl':
        # Load LLM_topks .pkl file
        LLM_topks_path = output_dir + LLM_topks_file_name
        with open(LLM_topks_path, 'rb') as f:
            og_LLM_topks = pickle.load(f)
    elif load_reply_strategy == 'txt':
        # Load LLM_topks .txt file
        samples = LoadLLMRepliesFromTextFiles(output_dir)
        if experiment_section == '3.2':
            preds, og_LLM_topks = parseLLMTopKsFromTxtFiles(samples, LLM_top_k, experiment_section=experiment_section)
        else:
            og_LLM_topks = parseLLMTopKsFromTxtFiles(samples, LLM_top_k, experiment_section=experiment_section)

    if experiment_section == '3.2':
        print("preds shape: ", len(preds))
        # create df of preds and actuals and name it accuracy_{acc}.csv where acc is integer accuracy
        preds = np.array([int(pred) for pred in preds])
        hidden_ys = np.load(output_dir + 'hidden_ys.pkl', allow_pickle=True)
        preds_df = pd.DataFrame({'preds': preds, 'hidden_ys': hidden_ys}, columns=['preds', 'hidden_ys'])
        acc = int(np.mean(preds == hidden_ys)*100)
        preds_df.to_csv(output_dir + f'accuracy_{acc}.csv', index=False)

        np.save(output_dir + 'preds.npy', preds)
    print("og_LLM_topks shape: ", len(og_LLM_topks))
    #print("og_LLM_topks", og_LLM_topks)

    # Remove unsolvable idxs from LLM_topks
    LLM_topks = copy.deepcopy(og_LLM_topks)
    LLM_topks = [LLM_topk for i, LLM_topk in enumerate(LLM_topks) if not unsolvable_idx[i]]

    # Remove bad replies from LLM_topks
    LLM_topks, orig_inds = removeBadReplies(LLM_topks, orig_inds, LLM_top_k)
    replies_df, unsolvable_idxs, bad_reply_idxs = constructReplies(eval_min_idx, eval_max_idx, og_LLM_topks,
                                                                   orig_inds, unsolvable_idx)
    print(f"{len(unsolvable_idxs)} Unsolvable Indices: ", unsolvable_idxs)
    print(f"{len(LLM_topks)} Good Replies:", orig_inds)
    print(f"{len(bad_reply_idxs)} Bad Replies: ", bad_reply_idxs)

    explanations         = makeFakeRankMagnitudesForFaithfulnessCalculation(LLM_topks, num_features)

    if eval_top_k == -1:
        eval_top_k = num_features

    # Calculate faithfulness
    FAs, RAs, PGUs, PGIs = calculateFaithfulness(model, explanations, inputs, eval_min_idx, len(explanations),
                                                 num_features, perturbation, perturb_num_samples, feature_types,
                                                 eval_top_k, calculateAUC)
    if calculateAUC:
        extra_str = 'AUC_k_' + str(eval_top_k)
    else:
        extra_str = '_eval_max_k_' + str(eval_top_k) + '_noAUC'


    if save_results:
        saveFaithfulnessMetrics(output_dir, FAs, RAs, PGUs, PGIs, orig_inds, replies_df,
                                output_file_write_type='w', extra_str=extra_str)
        saveParameters(output_dir, 'faithfulness_pipeline_config', config, extra_str)
    return getFaithfulnessMetricsString(model, FAs, RAs, PGUs, PGIs)

if __name__ == '__main__':
    runFaithfulnessPipeline()
