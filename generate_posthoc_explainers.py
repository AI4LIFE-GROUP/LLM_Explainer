from openxai.dataloader import return_loaders, get_feature_details
from openxai.Explainer import Explainer
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from faithulness_util import saveFaithfulnessMetrics, saveParameters
from faithulness_util import calculateFaithfulness
import torch
import pandas as pd
import os
import numpy as np
from utils import get_model_names, get_model_architecture, append_k, getExperimentID, DefineModel, shuffled_indices
bold = lambda x: '\033[1m' + x + '\033[0m'


def MakePostHocExplanations(post_hoc_explainer_name, SEED, inputs, model_name, data_name, output_dir,
                            load_explanations, explainer, labels=None):
    if load_explanations:
        exps = np.load(output_dir + 'test_' + data_name + '_' + model_name + '_' + post_hoc_explainer_name + '_explanations.npy')
    else:
        if post_hoc_explainer_name == 'lime':
            print(type(inputs))
            exps, _ = explainer.get_explanation(inputs.float(), seed=SEED, disable_tqdm=True)
        else:
            exps = explainer.get_explanation(inputs.float(), label=labels)
        np.save(output_dir + 'test_' + data_name + '_' + model_name + '_' + post_hoc_explainer_name + '_explanations.npy',
            exps.detach().numpy(), allow_pickle=False)
    return explainer, exps


load_explanations              = False
load_exp_dir                   = 'outputs/Explainers/20230825_002537_credit_ann_l/'  #if load_explatins is True, then load explanations from this directory. 1 model+dataset at a time (for now)
use_new_exp_id_for_final_table = True
exp_id_for_final_table         = '20240328_235213' # if use_new_exp_id_for_final_table is True, then use this exp_id for saving results to the final table
calculateAUC                   = True

SEED           = 0
algos          = ['shap', 'lime', 'sg', 'ig', 'itg', 'grad', 'random'] #, 'sg', 'ig', 'itg', 'shap', 'lime', 'random']
data_names     = ['compas']#['blood', 'adult', 'credit', 'compas']  # ', 'heloc']  # ['compas', 'adult', 'heloc']  # 'german', 'heloc', 'credit']
model_names    = ['ann_xl']#, 'lr']  # , 'ann_s', 'ann_m', 'ann_l', 'ann_xl']
base_model_dir = 'models/ClassWeighted/'
ks             = [3]
eval_min_idx   = 0
eval_max_idx   = 100


## Faithfulness metric perturbation class parameters
perturbation_mean            = 0.0
perturbation_std             = 0.1
perturbation_flip_percentage = np.sqrt((2/np.pi))*perturbation_std
perturb_num_samples          = 10000
categorical_features = {
    'compas': [],#[3, 4, 5],
    'adult': [],#[6, 7, 8, 9, 10, 11, 12],
    'credit': [],
    'blood': []
}

#LIME
kernel_width           = 0.75
std_LIME               = 0.1
mode                   = 'tabular'
sample_around_instance = True
n_samples_LIME         = 1000#16
discretize_continuous  = False

# grad
absolute_value = True

# Smooth grad
n_samples_SG = 100#16
std_SG       = 0.005

# Integrated gradients
method             = 'gausslegendre'
multiply_by_inputs = False
n_steps            = 50#16

#SHAP
n_samples = 500#16

#Make pandas dataframes to save faithfulness metrics for each dataset and model. The rows are the explainer,
# there will be one table per dataset and each column corresponds to each model's faithfulness score
# (FA, RA, PGI, and PGU)
# Make a dict of pandas dataframes for each model for each dataset
LR_metrics = append_k(ks, ["FA", "RA", "PGU", "PGI"])
ANN_metrics = append_k(ks, ["PGU", "PGI"])

faithfulness_dicts = {
    "compas": {
        # "lr": pd.DataFrame(index=algos, columns=LR_metrics),
        "ann_xl": pd.DataFrame(index=algos, columns=ANN_metrics),
    },
    "adult": {
        "lr": pd.DataFrame(index=algos, columns=LR_metrics),
        "ann_l": pd.DataFrame(index=algos, columns=ANN_metrics),
    },
    "german": {
        "lr": pd.DataFrame(index=algos, columns=LR_metrics),
        "ann_l": pd.DataFrame(index=algos, columns=ANN_metrics),
    },
    "heloc": {
        "lr": pd.DataFrame(index=algos, columns=LR_metrics),
        "ann_l": pd.DataFrame(index=algos, columns=ANN_metrics),
    },
    "credit": {
        "lr": pd.DataFrame(index=algos, columns=LR_metrics),
        "ann_l": pd.DataFrame(index=algos, columns=ANN_metrics),
    },
    "blood": {
        "lr": pd.DataFrame(index=algos, columns=LR_metrics),
        "ann_l": pd.DataFrame(index=algos, columns=ANN_metrics),
    },
    "beauty": {
        "lr": pd.DataFrame(index=algos, columns=LR_metrics),
        "ann_l": pd.DataFrame(index=algos, columns=ANN_metrics),
    }
}

# Loop over datasets
if use_new_exp_id_for_final_table:
    exp_id_for_final_table = getExperimentID()

for d, data_name in enumerate(data_names):
    print(bold('Data:'), data_name)
        
    # load data
    download_data = False if data_name in ['compas', 'blood'] else True
    loader_train, loader_val, loader_test = return_loaders(data_name=data_name, download=download_data, scaler='minmax')
    
    num_feats     = loader_train.dataset.X.shape[1]
    feature_types = ['c']*num_feats
    discrete_inds = np.array(categorical_features[data_name])
    for i, idx in enumerate(discrete_inds):
        feature_types[discrete_inds[i]] = 'd'
        _, feature_names, conversion, suffixes = get_feature_details(data_name, n_round=5)
    
    X_train, y_train = loader_train.dataset.data, loader_train.dataset.targets.to_numpy()
    X_val, y_val     = loader_val.dataset.data, loader_val.dataset.targets.to_numpy()
    X_test, y_test   = loader_test.dataset.data, loader_test.dataset.targets.to_numpy()

    if mode == 'text':
        X_train_sentences = loader_train.dataset.sentences
        X_val_sentences = loader_val.dataset.sentences
        X_test_sentences = loader_test.dataset.sentences

    # Loop over models
    for model_name in model_names:
        print(bold('Model:'), model_name)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        exp_id     = getExperimentID()
        output_dir = './outputs/Explainers/'+exp_id+'_'+data_name+'_'+model_name+'/'
        
        if not os.path.isdir(output_dir):  # If folder doesn't exist, then create it.
            os.makedirs(output_dir)
        
        # Define the model
        model_dir, model_file_name                          = get_model_names(model_name, data_name, base_model_dir)
        dim_per_layer_per_MLP, activation_per_layer_per_MLP = get_model_architecture(model_name)
        model                                               = DefineModel(num_feats, model_name, dim_per_layer_per_MLP,
                                                                          activation_per_layer_per_MLP)
        model.load_state_dict(torch.load(model_dir + model_file_name))
        model.eval()
        
        train_data_tensor = torch.FloatTensor(loader_train.dataset.data)

        # For benchmarking post hoc explanation methods on 1000 testing points
        inputs            = torch.FloatTensor(loader_test.dataset.data)
        labels            = torch.LongTensor(loader_test.dataset.targets.to_numpy())

        inputs = inputs[eval_min_idx:eval_max_idx]
        labels = labels[eval_min_idx:eval_max_idx]

        param_dict_lime = dict()
        param_dict_lime['dataset_tensor']         = train_data_tensor
        param_dict_lime['kernel_width']           = kernel_width
        param_dict_lime['std']                    = std_LIME
        param_dict_lime['mode']                   = mode
        param_dict_lime['sample_around_instance'] = sample_around_instance
        param_dict_lime['n_samples']              = n_samples_LIME
        param_dict_lime['discretize_continuous']  = discretize_continuous
        param_dict_lime['categorical_features']   = categorical_features[data_name]

        param_dict_grad                   = dict()
        param_dict_grad['absolute_value'] = absolute_value

        param_dict_sg                       = dict()
        param_dict_sg['n_samples']          = n_samples_SG
        param_dict_sg['standard_deviation'] = std_SG

        param_dict_ig                       = dict()
        param_dict_ig['method']             = method
        param_dict_ig['multiply_by_inputs'] = multiply_by_inputs
        param_dict_ig['baseline']           = torch.mean(train_data_tensor, dim=0).reshape(1, -1).float()
        param_dict_ig['n_steps']            = n_steps

        param_dict_shap              = dict()
        param_dict_shap['n_samples'] = n_samples

        param_dicts = {'lime': param_dict_lime, 'grad': param_dict_grad, 'sg': param_dict_sg, 'ig': param_dict_ig,
                       'shap': param_dict_shap, 'itg': dict(), 'random': dict()}

        # Make and save explanations or load them in
        explainers   = []
        explanations = []
        params       = []
        for algo in algos:
            if algo == 'random':
                exps = shuffled_indices(eval_max_idx - eval_min_idx, num_feats)
                np.save(output_dir + 'test_' + data_name + '_' + model_name + '_random_explanations.npy', exps, allow_pickle=False)
                explainer = 'random explainer'
                params.append(dict())
            else:
                explainer = Explainer(method=algo, model=model, dataset_tensor=train_data_tensor,
                                      param_dict=param_dicts[algo])
                params.append(param_dicts[algo])

            if not algo == 'random':
                explainer, exps = MakePostHocExplanations(algo, SEED, inputs, model_name, data_name, output_dir,
                                                          load_explanations, explainer, labels)

            explainers.append(explainer)
            explanations.append(exps)
        
        # Evaluate explanations
        perturbation = NormalPerturbation("tabular", mean=perturbation_mean,
                                          std_dev=perturbation_std, flip_percentage=perturbation_flip_percentage)

        for explainer, explanation_x, algo, param in zip(explainers, explanations, algos, params):
            print("explainer", explainer)
            print("algo", algo)
            # check if tensor
            if not isinstance(explanation_x, torch.Tensor):
                explanation_x = torch.tensor(explanation_x)
            for k in ks:
                # Calculate faithfulness
                FAs, RAs, PGUs, PGIs = calculateFaithfulness(model, explanation_x, inputs, eval_min_idx,
                                                             len(explanation_x), num_feats, perturbation,
                                                             perturb_num_samples, feature_types, k, calculateAUC)
                if calculateAUC:
                    extra_str = 'AUC_k_' + str(k)
                else:
                    extra_str = '_eval_max_k_' + str(k) + '_noAUC'

                N_samps = len(PGUs)
                if hasattr(model, 'return_ground_truth_importance'):
                    FA_metric = str(round(np.mean(FAs), 3)) + '+/-' + str(round(np.std(FAs) / np.sqrt(N_samps), 3))
                    RA_metric = str(round(np.mean(RAs), 3)) + '+/-' + str(round(np.std(RAs) / np.sqrt(N_samps), 3))
                PGU_metric = str(round(np.mean(PGUs), 3)) + '+/-' + str(round(np.std(PGUs) / np.sqrt(N_samps), 3))
                PGI_metric = str(round(np.mean(PGIs), 3)) + '+/-' + str(round(np.std(PGIs) / np.sqrt(N_samps), 3))

                # Store the FA_metric into the column
                if hasattr(model, 'return_ground_truth_importance'):
                    faithfulness_dicts[data_name][model_name].loc[algo, "FA_"+str(k)] = FA_metric
                    faithfulness_dicts[data_name][model_name].loc[algo, "RA_"+str(k)] = RA_metric
                faithfulness_dicts[data_name][model_name].loc[algo, "PGU_"+str(k)] = PGU_metric
                faithfulness_dicts[data_name][model_name].loc[algo, "PGI_"+str(k)] = PGI_metric

                saveFaithfulnessMetrics(output_dir, FAs, RAs, PGUs, PGIs, None, extra_str='_'+model_name+'_'+data_name+'_'+algo+'_k_'+str(k), replies_df = None)
                saveParameters(output_dir, 'faithfulness_config_'+model_name+'_'+data_name+'_'+algo+'_k_'+str(k), param)
                saveParameters('outputs/Explainers/', exp_id_for_final_table + '_faithfulness_dicts', faithfulness_dicts)

                # convert dict of dicts containing pandas dataframes, to one large pandas df and save to csv

                # Code courtesy of GPT4
                # Initialize an empty list to collect DataFrames
                df_list = []

                # Iterate through the main dictionary
                for main_key, sub_dict in faithfulness_dicts.items():
                    # Iterate through the sub-dictionary
                    for sub_key, df in sub_dict.items():
                        # Create a new DataFrame with added columns for the dictionary keys and the reset index
                        new_df = df.reset_index().copy()
                        new_df['Dataset'] = main_key
                        new_df['Model']   = sub_key
                        new_df.rename(columns={'index': 'Method'}, inplace=True)

                        # Add the new DataFrame to the list
                        df_list.append(new_df)

                # Concatenate all the new DataFrames into a single DataFrame
                final_df = pd.concat(df_list, ignore_index=True)

                # Save the final DataFrame to a CSV file
                final_df.to_csv('./outputs/Explainers/'+exp_id_for_final_table+'_faithfulness_dicts.csv', index=False)

