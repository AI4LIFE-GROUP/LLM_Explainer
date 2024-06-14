# call pipeline and evaluate

from utils import _load_config
from LLM_PostHocPipeline import Pipeline
from FaithfulnessPipeline import runFaithfulnessPipeline


datasets   = ['yelp', 'amazon_1000', 'imdb']  # 'blood', 'adult', 'credit', 'compas']
std_devs   = 0.1
max_ks     = [3]
models     = ['text_ann']#, 'ann_l']
n_shots    = [16]
prompt_ids = ['senti_classif_remove_words_pgicl']  # , 'io1-topk', 'io1-topk']

# Load pipeline config/prompts
config  = _load_config('LLM_pipeline_config.json')
prompts = _load_config('prompts.json')

for model in models:
    config['model_name'] = model
    for dataset in datasets:
        config['data_name'] = dataset
        for n_shot in n_shots:
            config['n_shot'] = n_shot
            for i, max_k in enumerate(max_ks):
                for prompt_id in prompt_ids:
                    if dataset == 'blood':
                        max_k = 4
                    print(prompt_ids[i])
                    config['prompt_params']['prompt_ID'] = prompt_id
                    config['prompt_params']['k']         = max_k

                    print(config)
                    pipeline           = Pipeline(config = config, prompts = prompts)
                    folder_name_exp_id = pipeline.run()

                    # Load faithfulness config
                    # faithfulness_config = _load_config('faithfulness_config.json')

                    # faithfulness_config['data_name']           = config['data_name']
                    # faithfulness_config['model_name']          = config['model_name']
                    # faithfulness_config['data_scaler']         = config['data_scaler']
                    # faithfulness_config['base_model_dir']      = config['base_model_dir']
                    # faithfulness_config['eval_min_idx']        = config['eval_min_idx']
                    # faithfulness_config['eval_max_idx']        = config['eval_max_idx']
                    # faithfulness_config['output_dir']          = "./outputs/LLM_QueryAndReply/" + folder_name_exp_id + "/"
                    # faithfulness_config['LLM_topks_file_name'] = ('n_shot_' + str(config['n_shot']) + '_' + \
                    #                                              config['data_name'] + '_' + config['model_name'] + \
                    #                                               '_LLM_topK.pkl')

                    # faithfulness_config['eval_max_k'] = max_k
                    # faithfulness_config['LLM_top_k']  = config['prompt_params']['k']
                    # runFaithfulnessPipeline(config = faithfulness_config)
