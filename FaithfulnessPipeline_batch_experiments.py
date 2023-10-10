import warnings
warnings.filterwarnings("ignore")

from FaithfulnessPipeline import runFaithfulnessPipeline
from utils import _load_config

experiment_dir = 'outputs/LLM_QueryAndReply/'

# add '/' at end of each experiment directory
#assumes data name is the second to last part of the experiment ID
#assumes model name is the last part of the experiment ID
experiment_IDs = [
# '20230926_135433_gpt-4_perturb_nshot16_k5_prompt-io1-topk_adult_lr/',
'20231009_221308_gpt-4_perturb_nshot17_k5_prompt-pfpe2-topk_adult_lr/',
]

calculateAUC       = True
experiment_section = "3.2"

eval_top_ks = [3]#[1, 2, 3, 4, 5]

metrics_str_per_experiment = []
for i, experiment_ID in enumerate(experiment_IDs):
    for eval_top_k in eval_top_ks:

        if 'ann_' in experiment_ID:
            model_name = 'ann_' + experiment_ID.split('_')[-1].replace('/', '')
            data_name = experiment_ID.split('_')[-3]
        else:
            data_name = experiment_ID.split('_')[-2]
            model_name = 'lr'

        if data_name == 'blood' and eval_top_k > 4:
            continue
        if data_name == 'blood':
            LLM_top_k = 4
        else:
            LLM_top_k = 5

        print('\nmax_k: ', eval_top_k)
        print('experiment_ID: ', experiment_ID)
        # Load faithfulness config
        # SET THE REMAINING HYPERPARAMS IN THE FAITHFULNESS CONFIG FILE
        config                       = _load_config('faithfulness_config.json')
        config['data_name']          = data_name
        config['model_name']         = model_name
        print(config['data_name'], config['model_name'])
        config['eval_top_k']         = eval_top_k
        config['output_dir']         = experiment_dir + experiment_ID
        config['LLM_top_k']          = LLM_top_k
        config['calculateAUC']       = calculateAUC
        config['experiment_section'] = experiment_section

        metrics_str = runFaithfulnessPipeline(config)
        metrics_str_per_experiment.append(metrics_str)

for metrics_str in metrics_str_per_experiment:
    print(metrics_str)
