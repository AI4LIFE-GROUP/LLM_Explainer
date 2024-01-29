import subprocess
import os

# List of directories
directories = [
# 3.1, delta_format = True, LR
'20240118_145133_gpt-4_perturb_nshot16_k4_prompt-explain_blood_lr/',
'20240118_153538_gpt-4_perturb_nshot16_k5_prompt-explain_compas_lr/',
'20240118_162128_gpt-4_perturb_nshot16_k5_prompt-explain_credit_lr/',
'20240118_165912_gpt-4_perturb_nshot16_k5_prompt-explain_adult_lr/',
# 3.1, delta_format = True, ANN_L
'20240119_234329_gpt-4_perturb_nshot16_k4_prompt-explain_blood_ann_l/',
'20240120_001133_gpt-4_perturb_nshot16_k5_prompt-explain_compas_ann_l/',
'20240120_003927_gpt-4_perturb_nshot16_k5_prompt-explain_credit_ann_l/',
'20240120_082839_gpt-4_perturb_nshot16_k5_prompt-explain_adult_ann_l/',

# 3.2, delta_format = True, LR
'20240120_124557_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_lr/',
'20240120_135959_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_lr/',
'20240120_143646_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_lr/',
'20240120_152851_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_lr/',
# 3.2, delta_format = True, ANN_L
'20240120_180956_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_ann_l/',
'20240120_183849_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_ann_l/',
'20240120_191201_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_ann_l/',
'20240120_195646_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_ann_l/',

# 3.1, delta_format = False, LR
'20240120_225834_gpt-4_perturb_nshot16_k4_prompt-explain_blood_lr/',
'20240120_231536_gpt-4_perturb_nshot16_k5_prompt-explain_compas_lr/',
'20240120_233701_gpt-4_perturb_nshot16_k5_prompt-explain_credit_lr/',
'20240120_235717_gpt-4_perturb_nshot16_k5_prompt-explain_adult_lr/',
# 3.1, delta_format = False, ANN_L
'20240121_001859_gpt-4_perturb_nshot16_k4_prompt-explain_blood_ann_l/',
'20240121_004633_gpt-4_perturb_nshot16_k5_prompt-explain_compas_ann_l/',
'20240121_011037_gpt-4_perturb_nshot16_k5_prompt-explain_credit_ann_l/',
'20240121_013351_gpt-4_perturb_nshot16_k5_prompt-explain_adult_ann_l/',

# 3.2, delta_format = False, LR
'20240121_102350_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_lr/',
'20240121_105646_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_lr/',
'20240121_113348_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_lr/',
'20240121_122407_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_lr/',
# 3.2, delta_format = False, ANN_L
'20240121_130412_gpt-4_perturb_nshot16_k4_prompt-explain_with_instructions_blood_ann_l/',
'20240121_133431_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_compas_ann_l/',
'20240121_141224_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_credit_ann_l/',
'20240121_150712_gpt-4_perturb_nshot16_k5_prompt-explain_with_instructions_adult_ann_l/',
]

base_path = os.getcwd()

for dir_name in directories:
    # Construct the full path to the directory
    full_path = os.path.join(base_path, 'outputs', 'LLM_QueryAndReply', dir_name)

    # Check if the directory exists
    if os.path.isdir(full_path):
        # Change to the directory
        os.chdir(full_path)

        # Find all files starting with 'faithfulness' or 'Faithfulness'
        for file in os.listdir(full_path):
            if file.startswith('faithfulness') or file.startswith('Faithfulness'):
                # Add the file
                subprocess.run(['git', 'add', file])

        # Commit the changes
        commit_message = f"Committing faithfulness files in {dir_name}"
        subprocess.run(['git', 'commit', '-m', commit_message])

        # Push the commit
        subprocess.run(['git', 'push', 'origin', 'main'])
    else:
        print(f"Directory not found: {full_path}")

# Return to the original directory if needed
# os.chdir(original_directory)
