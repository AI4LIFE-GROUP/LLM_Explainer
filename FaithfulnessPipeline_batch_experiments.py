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
# 3.0, delta_format = True, LR
#'20240117_175146_gpt-4_perturb_nshot16_k5_prompt-logprob_credit_lr/',
# '20240117_174850_gpt-4_perturb_nshot16_k5_prompt-logprob_adult_lr/',
# '20240117_230441_gpt-4_perturb_nshot16_k4_prompt-logprob_blood_lr/',
# '20240117_230734_gpt-4_perturb_nshot16_k5_prompt-logprob_compas_lr/',
# 3.0, delta_format = True, ANN_L
# '20240117_235624_gpt-4_perturb_nshot16_k4_prompt-logprob_blood_ann_l/',
# '20240118_000135_gpt-4_perturb_nshot16_k5_prompt-logprob_compas_ann_l/',
# '20240118_000453_gpt-4_perturb_nshot16_k5_prompt-logprob_credit_ann_l/',
# '20240118_000745_gpt-4_perturb_nshot16_k5_prompt-logprob_adult_ann_l/'
# 3.0, delta_format = False, LR
# '20240118_113244_gpt-4_perturb_nshot16_k4_prompt-logprob_blood_lr/',
# '20240118_114514_gpt-4_perturb_nshot16_k5_prompt-logprob_compas_lr/',
# '20240118_114835_gpt-4_perturb_nshot16_k5_prompt-logprob_credit_lr/',
# '20240118_115211_gpt-4_perturb_nshot16_k5_prompt-logprob_adult_lr/',

# 3.1, delta_format = True, LR
# '20240118_145133_gpt-4_perturb_nshot16_k4_prompt-explain_blood_lr/',
# '20240118_153538_gpt-4_perturb_nshot16_k5_prompt-explain_compas_lr/',
# '20240118_162128_gpt-4_perturb_nshot16_k5_prompt-explain_credit_lr/',
# '20240118_165912_gpt-4_perturb_nshot16_k5_prompt-explain_adult_lr/',
# 3.1, delta_format = True, ANN_L
# '20240119_234329_gpt-4_perturb_nshot16_k4_prompt-explain_blood_ann_l/',
# '20240120_001133_gpt-4_perturb_nshot16_k5_prompt-explain_compas_ann_l/',
# '20240120_003927_gpt-4_perturb_nshot16_k5_prompt-explain_credit_ann_l/',
# '20240120_082839_gpt-4_perturb_nshot16_k5_prompt-explain_adult_ann_l/',

# 3.2, delta_format = True, LR
# '20240120_124557_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_lr/',
# '20240120_135959_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_lr/',
# '20240120_143646_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_lr/',
# '20240120_152851_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_lr/'
# 3.2, delta_format = True, ANN_L
# '20240120_180956_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_ann_l/',
# '20240120_183849_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_ann_l/',
# '20240120_191201_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_ann_l/',
# '20240120_195646_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_ann_l/',

# 3.1, delta_format = False, LR
# '20240120_225834_gpt-4_perturb_nshot16_k4_prompt-explain_blood_lr/',
# '20240120_231536_gpt-4_perturb_nshot16_k5_prompt-explain_compas_lr/',
# '20240120_233701_gpt-4_perturb_nshot16_k5_prompt-explain_credit_lr/',
# '20240120_235717_gpt-4_perturb_nshot16_k5_prompt-explain_adult_lr/',
# 3.1, delta_format = False, ANN_L
# '20240121_001859_gpt-4_perturb_nshot16_k4_prompt-explain_blood_ann_l/',
# '20240121_004633_gpt-4_perturb_nshot16_k5_prompt-explain_compas_ann_l/',
# '20240121_011037_gpt-4_perturb_nshot16_k5_prompt-explain_credit_ann_l/',
# '20240121_013351_gpt-4_perturb_nshot16_k5_prompt-explain_adult_ann_l/',

# 3.2, delta_format = False, LR
# '20240121_102350_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_lr/',
# '20240121_105646_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_lr/',
# '20240121_113348_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_lr/',
# '20240121_122407_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_lr/',
# 3.2, delta_format = False, ANN_L
# '20240121_130412_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_ann_l/',
# '20240121_133431_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_ann_l/',
# '20240121_141224_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_ann_l/',
# '20240121_150712_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_ann_l/',

# 3.1, delta_format = True, LR/ANN_L, no context
# '20240122_080814_gpt-4_perturb_nshot16_k5_prompt-explain_no_context_adult_lr/',
# '20240122_084447_gpt-4_perturb_nshot16_k5_prompt-explain_no_context_adult_ann_l/',
# 3.2, delta_format = True, LR/ANN_L, no context
# '20240122_105843_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_no_context_adult_lr/',
# '20240122_132435_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_no_context_adult_ann_l/',

# 3.1, delta_format = True, LR/ANN_L, no CoT
# '20240122_143920_gpt-4_perturb_nshot16_k5_prompt-explain_adult_lr/',
# '20240122_144613_gpt-4_perturb_nshot16_k5_prompt-explain_adult_ann_l/',
# 3.2, delta_format = True, LR/ANN_L, no CoT
# '20240122_144230_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_lr/',
# '20240122_144902_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_ann_l/',

# 3.1, delta_format = True, LR, n_shot = 4, 8, 12, 32
# '20240123_102140_gpt-4_perturb_nshot4_k5_prompt-explain_adult_lr/',
# '20240123_133337_gpt-4_perturb_nshot8_k5_prompt-explain_adult_lr/',
# '20240123_171030_gpt-4_perturb_nshot12_k5_prompt-explain_adult_lr/',
# '20240123_201454_gpt-4_perturb_nshot32_k5_prompt-explain_adult_lr/',

# 3.2, delta_format = True, LR, n_shot = 4, 8, 12, 32
# '20240123_110546_gpt-4_perturb_nshot4_k5_prompt-explain_with_instructions_adult_lr/',
# '20240123_141551_gpt-4_perturb_nshot8_k5_prompt-explain_with_instructions_adult_lr/',
# '20240123_175430_gpt-4_perturb_nshot12_k5_prompt-explain_with_instructions_adult_lr/',
# '20240123_205159_gpt-4_perturb_nshot32_k5_prompt-explain_with_instructions_adult_lr/',

# 3.1, delta_format = True, ANN_L, n_shot = 4, 8, 12, 32
# '20240123_230204_gpt-4_perturb_nshot4_k5_prompt-explain_adult_ann_l/',
# '20240124_013415_gpt-4_perturb_nshot8_k5_prompt-explain_adult_ann_l/',
# '20240124_125414_gpt-4_perturb_nshot12_k5_prompt-explain_adult_ann_l/',
# '20240124_160111_gpt-4_perturb_nshot32_k5_prompt-explain_adult_ann_l/',

# 3.2, delta_format = True, ANN_L, n_shot = 4, 8, 12, 32
# '20240123_233218_gpt-4_perturb_nshot4_k5_prompt-explain_with_instructions_adult_ann_l/',
# '20240124_021522_gpt-4_perturb_nshot8_k5_prompt-explain_with_instructions_adult_ann_l/',
# '20240124_134406_gpt-4_perturb_nshot12_k5_prompt-explain_with_instructions_adult_ann_l/',
# '20240124_164010_gpt-4_perturb_nshot32_k5_prompt-explain_with_instructions_adult_ann_l/',

# 3.1/3.2, no context, blood, LR
# '20240124_185504_gpt-4_perturb_nshot16_k4_prompt-explain_no_context_blood_lr/',
# '20240124_191930_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_no_context_blood_lr/',
# 3.1/3.2, no context, blood, ANN_L
# '20240124_195618_gpt-4_perturb_nshot16_k4_prompt-explain_no_context_blood_ann_l/',
# '20240124_202108_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_no_context_blood_ann_l/',

# CoT = False, blood, LR/ANN_L, delta_format = True
# '20240124_215821_gpt-4_perturb_nshot16_k4_prompt-explain_blood_lr/',  # CoT = False
# '20240124_220033_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_lr/',  # CoT = False
# '20240124_220423_gpt-4_perturb_nshot16_k4_prompt-explain_blood_ann_l/',  # CoT = False
# '20240124_220651_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_ann_l/',  # CoT = False

# 3.1, delta_format = True, LR, n_shot = 4, 8, 12, 32, blood
# '20240125_020914_gpt-4_perturb_nshot4_k4_prompt-explain_blood_lr/',
# '20240125_035408_gpt-4_perturb_nshot8_k4_prompt-explain_blood_lr/',
# '20240125_053823_gpt-4_perturb_nshot12_k4_prompt-explain_blood_lr/',
# '20240125_072935_gpt-4_perturb_nshot32_k4_prompt-explain_blood_lr/',

# 3.2, delta_format = True, LR, n_shot = 4, 8, 12, 32, blood
# '20240125_025511_gpt-4_perturb_nshot4_k4_prompt-explain_with_instructions_blood_lr/',
# '20240125_043822_gpt-4_perturb_nshot8_k4_prompt-explain_with_instructions_blood_lr/',
# '20240125_062431_gpt-4_perturb_nshot12_k4_prompt-explain_with_instructions_blood_lr/',
# '20240125_081444_gpt-4_perturb_nshot32_k4_prompt-explain_with_instructions_blood_lr/',

# 3.1, 3.2, delta_format = True, LR/ANN_L, blood, gpt 3.5
# '20240125_185840_gpt-3.5-turbo-1106_perturb_nshot16_k4_prompt-explain_blood_lr/',
# '20240125_191121_gpt-3.5-turbo-1106_perturb_nshot16_k4_prompt-explain_with_instructions_blood_lr/',
# '20240125_192820_gpt-3.5-turbo-1106_perturb_nshot16_k4_prompt-explain_blood_ann_l/',
# '20240125_193026_gpt-3.5-turbo-1106_perturb_nshot16_k4_prompt-explain_with_instructions_blood_ann_l/',

# 3.1, 3.2, delta_format = True, LR/ANN_L, adult, gpt 3.5
# '20240125_191721_gpt-3.5-turbo-1106_perturb_nshot16_k5_prompt-explain_adult_lr/',
# '20240125_192103_gpt-3.5-turbo-1106_perturb_nshot16_k5_prompt-explain_with_instructions_adult_lr/',
# '20240125_193547_gpt-3.5-turbo-1106_perturb_nshot16_k5_prompt-explain_adult_ann_l/',
# '20240125_193925_gpt-3.5-turbo-1106_perturb_nshot16_k5_prompt-explain_with_instructions_adult_ann_l/',

# 3.1, delta_format = True, LR/ANN_L, blood, predict then explain vs predict
# '20240125_212043_gpt-4_perturb_nshot16_k4_prompt-predict_then_explain_blood_lr/',
#'20240118_145133_gpt-4_perturb_nshot16_k4_prompt-explain_blood_lr/',
# '20240125_222917_gpt-4_perturb_nshot16_k4_prompt-predict_then_explain_blood_ann_l/',
#'20240119_234329_gpt-4_perturb_nshot16_k4_prompt-explain_blood_ann_l/',

# 3.1, delta_format = True, LR/ANN_L, adult, predict then explain vs predict
# '20240125_221140_gpt-4_perturb_nshot16_k5_prompt-predict_then_explain_adult_lr/',
#'20240118_165912_gpt-4_perturb_nshot16_k5_prompt-explain_adult_lr/',
# '20240125_224008_gpt-4_perturb_nshot16_k5_prompt-predict_then_explain_adult_ann_l/',
#'20240120_082839_gpt-4_perturb_nshot16_k5_prompt-explain_adult_ann_l/',
]

calculateAUC       = False
experiment_section = "3.2"

eval_top_ks = [1, 2, 3, 4, 5]

metrics_str_per_experiment = []
for i, experiment_ID in enumerate(experiment_IDs):
    if 'ann_' in experiment_ID:
        model_name = 'ann_' + experiment_ID.split('_')[-1].replace('/', '')
        data_name = experiment_ID.split('_')[-3]
    else:
        data_name = experiment_ID.split('_')[-2]
        model_name = 'lr'
    for eval_top_k in eval_top_ks:
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
