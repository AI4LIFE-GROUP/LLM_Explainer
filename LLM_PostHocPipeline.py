
"""
Description:    This script runs the LLM post-hoc pipeline for a given dataset and model.
                The pipeline consists of the following steps:
                    1. Load dataset and model
                    2. Generate prompt for LLM
                    3. Get explanations from LLM
                    4. Save results
                The script is organized as follows:
                    1. Import packages and functions
                    2. Define additional parameters/functions
                    3. Define parameters from config file
                    4. Run pipeline
                    5. Save results and config
"""

# Package Imports
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import string
import numpy as np
from utils import _load_config, SaveExperimentInfo, loadOpenAPIKeyFromFile
from utils import get_k_words, get_model_names, get_model_architecture

# XAI Imports
from openxai.dataloader import return_loaders, get_feature_details
from openxai.LoadModel import DefineModel

# LLM Imports
from llms.query import getExperimentID, SaveLLMQueryInfo
from llms.icl import ConstantICL, PerturbICL
from llms.response import processGPTReply, RobustQueryGPT
from llms.prompt import Prompt

bold    = lambda x: '\033[1m' + x + '\033[0m'
letters = [string.ascii_uppercase[i] for i in range(26)]    # A, B, C, ..., Z

class Pipeline:
    def __init__(self, config = None, prompts = None, do_setup = True):
        # Load config/prompts files
        self.config  = _load_config('LLM_pipeline_config.json') if config is None else config
        self.prompts = _load_config('prompts.json') if prompts is None else prompts

        if do_setup:
            self.load_dataset_and_model()
            self.generate_prompt_instance()
            self.generate_icl_sampler_instance()

    def load_dataset_and_model(self):
        # Read config
        self.data_name      = self.config['data_name']
        self.data_scaler    = self.config['data_scaler']
        self.model_name     = self.config['model_name']
        self.base_model_dir = self.config['base_model_dir']
        self.model_dir, self.model_file_name = get_model_names(self.model_name,
                                                               self.data_name,
                                                               self.base_model_dir)

        # Load dataset
        download_data                         = False if self.data_name in ['compas', 'blood'] else True
        loader_train, loader_val, loader_test = return_loaders(data_name=self.data_name, download=download_data,
                                                               scaler=self.data_scaler)

        self.X_train, self.y_train = loader_train.dataset.data, loader_train.dataset.targets.to_numpy()
        self.X_val, self.y_val     = loader_val.dataset.data, loader_val.dataset.targets.to_numpy()
        self.X_test, self.y_test   = loader_test.dataset.data, loader_test.dataset.targets.to_numpy()
        self.num_features          = self.X_train.shape[1]

        # Load model
        input_size                                          = loader_train.dataset.get_number_of_features()
        dim_per_layer_per_MLP, activation_per_layer_per_MLP = get_model_architecture(self.model_name)
        self.model                                          = DefineModel(self.model_name, input_size,
                                                                          dim_per_layer_per_MLP,
                                                                          activation_per_layer_per_MLP)
        self.model.load_state_dict(torch.load(self.model_dir + self.model_file_name))
        self.model.eval()

        # Store test predictions
        preds = self.model.predict(torch.tensor(self.X_test).float())
        if self.config['prompt_params']['use_soft_preds']:
            self.preds = preds[:, 1]
        else:
            self.preds = np.argmax(preds, axis=1)
    
    def generate_prompt_instance(self):
        # Read config
        self.n_shot             = self.config['n_shot']
        prompt_params           = self.config['prompt_params']
        self.use_soft_preds     = prompt_params['use_soft_preds']
        self.rescale_soft_preds = prompt_params['rescale_soft_preds']
        self.n_round            = prompt_params['n_round']
        self.k                  = prompt_params['k']
        self.feature_types, self.feature_names, self.conversion, self.suffixes = get_feature_details(self.data_name,
                                                                                                     self.n_round)

        self.categorical_features = [i for i, f in enumerate(self.feature_types) if f == 'd']


        self.prompt = Prompt(feature_names=self.feature_names, input_str=prompt_params['input_str'],
                             output_str=prompt_params['output_str'], input_sep=prompt_params['input_sep'],
                             output_sep=prompt_params['output_sep'], feature_sep=prompt_params['feature_sep'],
                             value_sep=prompt_params['value_sep'], n_round=self.n_round,
                             hide_test_sample=prompt_params['hide_test_sample'],
                             hide_last_pred=prompt_params['hide_last_pred'],
                             hide_feature_details=prompt_params['hide_feature_details'],
                             conversion=self.conversion, suffixes=self.suffixes, feature_types=self.feature_types,
                             use_soft_preds=self.use_soft_preds, add_explanation=prompt_params['add_explanation'], prompt_id=prompt_params['prompt_ID'])
        k_words = get_k_words()
        self.k  = self.k if self.k != -1 else len(self.feature_types)

        prompt_variables = {
            'k_word': k_words[self.k-1],
            'num_features': self.num_features,
            'n_shot': self.n_shot,
            'final_feature': letters[self.num_features-1],
            'final_change': 'FILL IN ',
            'zero_change': ', '.join([letter + ': 0.000' for letter in letters[:self.num_features]]),
            'feat_words': [string.ascii_uppercase[i] for i in range(len(self.feature_names))] #[f'f{i+1}' for i in range(len(self.feature_names))]
            #'other_string_variables': 'access these in the config file using curly braces e.g. {k_word}',
        }
        prompt_ID                 = prompt_params['prompt_ID']
        prompt_info               = self.prompts[prompt_ID]
        self.pre_text             = prompt_info['pre_text'].format(**prompt_variables)
        self.post_text            = prompt_info['post_text'].format(**prompt_variables)
        self.mid_text             = prompt_info['mid_text'].format(**prompt_variables)
        self.reply_parse_strategy = 'COT' if 'COT' in prompt_ID else ''
        self.reply_parse_strategy = 'last' if 'io1' in prompt_ID else self.reply_parse_strategy

    def generate_icl_sampler_instance(self):
        # Read config
        self.icl_params          = self.config['icl_params']
        self.sampling_scheme     = self.icl_params['sampling_scheme']
        self.sampling_params     = self.config['sampling_params'][self.sampling_scheme]
        self.icl_seed            = self.icl_params['icl_seed']
        self.use_most_confident  = self.icl_params['use_most_confident']
        self.use_class_balancing = self.icl_params['use_class_balancing']
        self.sorting             = self.icl_params['sorting']

        sampling_classes = {'constant': ConstantICL, 'perturb': PerturbICL}
        params           = {'constant': {'model': self.model, 'sample_space': self.X_val},
                            'perturb': {'model': self.model, 'sample_space': None,
                                        'X_test': self.X_test, 'feature_types': self.feature_types}
                            }

        all_params       = {**params[self.sampling_scheme], **self.sampling_params}
        self.icl_sampler = sampling_classes[self.sampling_scheme](**all_params)
    
    def get_icl_samples(self, eval_idx, use_eval_as_seed=False):
        self.icl_sampler.eval_idx = eval_idx
        if use_eval_as_seed:
            self.icl_sampler.perturb_seed = eval_idx
        X_ICL, y_ICL = self.icl_sampler.sample(self.n_shot, self.icl_seed, self.use_soft_preds,
                                               self.use_most_confident, self.use_class_balancing, self.sorting)
        
        pred = self.preds[eval_idx]
        if self.use_soft_preds and self.rescale_soft_preds:
            y_max           = max(np.max(y_ICL), self.preds[eval_idx])
            y_min           = min(np.min(y_ICL), self.preds[eval_idx])
            y_ICL           = (y_ICL - y_min) / (y_max - y_min)
            pred            = (pred - y_min) / (y_max - y_min)

        # Create Prompt
        if self.config['prompt_params']['add_explanation']:
            self.num_explanations     = self.config['prompt_params']['num_explanations']
            self.explanation_sampling = self.config['icl_params']['explanation_sampling']
            self.rand_ind             = np.load(f'{self.data_name}_icl_explanation_index.npy')

            if self.explanation_sampling == 'random':
                rand_ind   = random.sample(range(0, self.num_explanations), self.n_shot)
                exp_x_test = self.X_val[self.rand_ind[rand_ind]]
                exp_exps   = self.explanations[rand_ind]
                exp_y_test = self.y_val[self.rand_ind[rand_ind]]
            elif self.explanation_sampling == 'balanced':
                ind_y_1    = self.rand_ind[self.y_val[self.rand_ind] == 1]
                ind_y_0    = self.rand_ind[self.y_val[self.rand_ind] == 0]
                rand_ind_1 = random.sample(ind_y_1.tolist(), self.n_shot//2)
                rand_ind_0 = random.sample(ind_y_0.tolist(), self.n_shot//2)
                rand_ind   = rand_ind_1 + rand_ind_0
                exp_x_test = self.X_val[rand_ind]
                exp_exps   = self.explanations[[np.where(self.rand_ind == i)[0][0] for i in rand_ind]]
                exp_y_test = self.y_val[rand_ind]
            else:
                print('Invalid choice!!')

        else:
            exp_x_test = None
            exp_exps   = None
            exp_y_test = None

        return X_ICL, y_ICL, pred, exp_x_test, exp_exps, exp_y_test

    def run(self):
        # Read config
        experiment_params        = self.config['experiment_params']
        openai_api_key_file_path = self.config['openai_api_key_file_path']
        LLM                      = self.config['LLM_name']
        temperature              = self.config['temperature']
        self.eval_min_idx        = self.config['eval_min_idx']
        eval_max_idx             = self.config['eval_max_idx']
        max_test_samples         = self.config['max_test_samples']
        self.eval_max_idx        = min(max_test_samples, len(self.y_test)) if eval_max_idx == -1 else eval_max_idx

        # Load openai's API key from text file
        api_key            = loadOpenAPIKeyFromFile(openai_api_key_file_path)
        folder_name_exp_id = (getExperimentID(**experiment_params) + '_' + LLM + '_' + self.sampling_scheme +
                              '_nshot' + str(self.n_shot) + '_k' + str(self.k) + '_prompt-' +
                              self.config['prompt_params']['prompt_ID']) + '_' + self.data_name + '_' + self.model_name

        # To test ICL with explanations
        if self.config['prompt_params']['add_explanation']:
            exp_method        = self.config['icl_params']['explanation_method']
            self.explanations = np.load(f'./OpenXAI_Explanations/icl_{self.data_name}_{self.model_name}_{exp_method}_explanations.npy')

        LLM_topks        = []
        hide_last_pred   = self.config['prompt_params']['hide_last_pred']
        hidden_ys        = None if not hide_last_pred else []
        perturb_seed     = self.config['sampling_params']['perturb']['perturb_seed']
        use_eval_as_seed = True if perturb_seed == "eval" else False
        for eval_idx in range(self.eval_min_idx, self.eval_max_idx):  # loop each test sample
            print(eval_idx, len(LLM_topks))

            # Get ICL samples (here, as it may depend on eval_idx)
            X_ICL, y_ICL, pred, exp_x_test, exp_exps, exp_y_test = self.get_icl_samples(eval_idx, use_eval_as_seed)

            if self.config['prompt_params']['prompt_ID'] in ['pe1', 'pe2', 'pe1-topk', 'pe2-topk']:
                pred = None
            elif self.config['prompt_params']['prompt_ID'].startswith('pfp'):
                X_ICL -= self.X_test[eval_idx]
                y_ICL -= pred
            # Generate prompt text
            prompt_outputs = self.prompt.create_prompt(X_train=X_ICL, y_train=y_ICL, x=self.X_test[eval_idx],
                                                       post_text=self.post_text, x_test=exp_x_test, explanations=exp_exps,
                                                       y_test=exp_y_test, pre_text=self.pre_text, y=pred, mid_text=self.mid_text)
            if hide_last_pred:
                prompt_text, last_y = prompt_outputs
                hidden_ys.append(last_y)
            else:
                prompt_text = prompt_outputs

            print("PROMPT:\n")
            print(prompt_text)

            # Query LLM
            reply, message = RobustQueryGPT(prompt_text, LLM, api_key, temperature)

            # Process the query reply. Keep only the feature names, remove extra punctuation
            print("REPLY:", reply)
            LLM_topks.append(processGPTReply(reply, self.reply_parse_strategy))

            if hide_last_pred:
                print("Last y:", last_y)

            # Save query info
            SaveLLMQueryInfo(folder_name_exp_id, LLM, self.model_name, self.data_name, temperature,
                             self.n_shot, eval_idx, self.k, message, prompt_text, reply, self.sampling_scheme)

        SaveExperimentInfo(self.config, folder_name_exp_id, self.n_shot, LLM, self.model_name, self.data_name, LLM_topks,
                           self.eval_min_idx, self.eval_max_idx, hidden_ys=hidden_ys)

        return folder_name_exp_id

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
    