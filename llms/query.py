"""Used to store LLMs and query the HuggingFace API"""

import requests
import time
import numpy as np
import datetime
import os

# Free inference APIs
LLM_APIS = {'bloom-176b': 'https://api-inference.huggingface.co/models/bigscience/bloom',  # largest open source LLM, 176B
            'flan-ul2-20b': 'https://api-inference.huggingface.co/models/google/flan-ul2',  # 20B, instruction-based
	        'gpt-neox-20b': 'https://api-inference.huggingface.co/models/EleutherAI/gpt-neox-20b', # 20B
            'flan-t5-xxl-11b': 'https://api-inference.huggingface.co/models/google/flan-t5-xxl',  # 11B
            'gpt-j-6b': 'https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6b',
            'gpt-3.5-turbo-0613': 'N/A'}  # 6B}

# Session Variables
API_TOKENS = [""]

# Experiment Functions

def getExperimentID(use_min=True, use_sec=True, use_microsec=False):
    date_info = datetime.datetime.now()
    microsec, sec, minute = '', '', ''
    if use_microsec:
        microsec = '_' + str(date_info.microsecond)[0:3]
    if use_sec:
        sec = str(date_info.second).rjust(2, '0')
    if use_min:
        minute = str(date_info.minute).rjust(2, '0')
    testID = '%d%02d%02d_%02d' % (date_info.year, date_info.month, date_info.day, date_info.hour)
    testID = testID + minute + sec + microsec
    return testID

# Save out query and reply
def SaveLLMQueryInfo(folder_name_exp_id, LLM, model_name, data_name,
                     temperature, n_shot, eval_idx, k, message,
                     prompt_text, reply, sampling_scheme):
    # Save to .txt
    output_dir = 'outputs/LLM_QueryAndReply/'+folder_name_exp_id
    if not os.path.isdir(output_dir):  # If folder doesn't exist, then create it.
        os.makedirs(output_dir)
    file_name  = str(eval_idx) + '_' + LLM + '_' + model_name.upper() + '_' + data_name + '_summary'

    fpth = os.path.join(output_dir, file_name+'.txt')
    paramTxt = open(fpth, 'w')

    paramTxt.write(file_name + '\n')
    paramTxt.write('temperature:\t\t'+str(temperature) + '\n')
    paramTxt.write('n_shot:\t\t\t'+str(n_shot) + '\n')
    paramTxt.write('explanation_mode:\t'+str(sampling_scheme) + '\n')
    # if explanation_mode == 'sample_and_vote_icl':
    #     paramTxt.write('num_trials_per_sample:\t'+str(num_trials_per_sample) + '\n')
    #paramTxt.write('ICL_idxs:\t\t'+str(ICL_idxs) + '\n')
    paramTxt.write('eval_idx:\t\t'+str(eval_idx) + '\n')
    paramTxt.write('LLM:\t\t\t'+LLM + '\n')
    paramTxt.write('k:\t\t\t'+str(k) + '\n')
    paramTxt.write('\nMESSAGE:\n' + str(message) + '\n\n')
    paramTxt.write('\nPROMPT_TEXT:\n'+prompt_text + '\n\n')
    paramTxt.write('\nREPLY:\n'+reply + '\n')
    paramTxt.close()


# Query
def query(LLM, prompt, delay=2, token_idx=0):
    """
    Query the HuggingFace API
    LLM: str, name of the LLM
    prompt: str, prompt to query the LLM with
    delay: int, delay in seconds between queries
    token_idx: int, index of API token to use
    """
    payload = {"inputs": prompt}
    API_URL = LLM_APIS[LLM]
    headers = {"Authorization": f"Bearer {API_TOKENS[token_idx]}"}
    time.sleep(delay)
    response = requests.post(API_URL, headers=headers, json=payload).json()

    # Error handling
    if 'error' in response:
        print('Error:', response['error'])
        if 'Rate limit' in response['error']:
            print('Waiting 1 minute and trying again with next token...')
            # Wait 1 minute and try again
            time.sleep(60)
            next_token_idx = (token_idx + 1) % len(API_TOKENS)
            return query(LLM, prompt, delay=delay, token_idx=next_token_idx)
        elif 'loading' in response['error']:
            print('Waiting 1 minute and trying again...')
            # Wait 1 minute and try again
            time.sleep(60)
            return query(LLM, prompt, delay=delay)
        elif 'Authorization' in response['error']:
            print('Please check your API token in llm_utils.py.')
            raise ValueError('Invalid API token.')
        else:
            return np.inf  # avoids breaking the llm_predictor loop

    return response[0]['generated_text']


# Example
# output = query('flan-t5-xxl-11b', 'The quick brown fox jumps over the lazy dog.')
