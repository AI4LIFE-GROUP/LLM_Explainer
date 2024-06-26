import logging
import copy
import numpy as np
import openai
import requests
from openai import OpenAI
import os
import string
import time
from .query import LLM_APIS
import re
letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]

huggingface_familys = {
    "llama": "meta-llama",
    "mistral": "mistralai",
}

huggingface_llms = {
    "llama-2-70b-chat-hf": "Llama-2-70b-chat-hf",
    "mixtral-8x7b-instruct": "Mixtral-8x7B-Instruct-v0.1",
}


def extract_pattern(string):
    pattern_regex = re.compile(r'(?i)(?<=The answer is:)\s*([A-Za-z,\s]+)')
    matches = re.search(pattern_regex, string)
    if matches:
        return matches.group(1)
    else:
        return None


def processGPTReply(reply, parse_strategy):
    # Process the query reply. Save only the feature names, remove extra punctuation
    if parse_strategy.upper() == 'COT':  # COT sometimes doesn't give the answer in the typical format
        reply_regex = extract_pattern(reply)  # do regex pattern matching
        if reply_regex == None:  # if regex didn't work, try a simpler method
            print("regex didn't work. Trying a simpler method.")
            temp_split = reply.upper().split('THE ANSWER IS:')  # try splitting by 'The answer is:'
            if len(temp_split) == 1:  # if splitting didn't work, then give up :(
                reply = 'None'
            else:
                reply = temp_split[1]
            # print("pre-processed reply", reply)
        else:  # do normal splitting
            reply = reply_regex
    elif parse_strategy.lower() == 'last':
        reply = reply.split('\n')[-1]
    reply_split = reply.split(',')
    LLM_topk    = [feat.split(':')[0].strip().replace('.','') for feat in reply_split]
    return LLM_topk

def queryGPT(prompt_text, LLM_name, api_key, temperature=0.0, max_tokens=None, return_tokens=False):
    client = OpenAI(api_key=api_key)
    message = [{"role": "user", "content": prompt_text}]  # put in config
    chat    = client.chat.completions.create(model=LLM_name, logprobs=False,# logit_bias=logit_bias,
                                             messages=message, temperature=temperature, max_tokens=max_tokens)
    reply   = chat.choices[0].message.content
        # print(reply)
        # count += 1
    
    #print(chat.usage.prompt_tokens, chat.usage.completion_tokens)
    # for token in chat.choices[0].logprobs:
    #     print(token)
    if return_tokens:
        return reply, message, chat.usage.prompt_tokens, chat.usage.completion_tokens
    return reply, message

def QueryHuggingFace(prompt_text, LLM, api_key):
    llm_family = huggingface_familys["llama" if "llama" in LLM.lower() else "mistral"]
    LLM = huggingface_llms[LLM]
    API_URL = f"https://api-inference.huggingface.co/models/{llm_family}/{LLM}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 250,  # maximum according to docs
            "return_full_text": False,
            "do_sample": False,
            "use_cache": False,
            "wait_for_model": True,
        }
    }
    attempts = 0
    while True:
        attempts += 1
        try:
            output = requests.post(API_URL, headers=headers, json=payload).json()
            return output[0]
        except Exception as e:
            print("ERROR! I'm going to sleep for " + str(attempts) + "s")
            print("Error:", str(e))
            try:
                print("output", output)
            except:
                print("No output")
            time.sleep(attempts+2)

def RobustQueryGPT(prompt_text, LLM, api_key, temperature=0.0, return_tokens=False):
    # Query (keep trying if we get an error)
    attempts = 0
    try_query = True
    while try_query:
        attempts += 1
        try:
            reply, message, prompt_tokens, completion_tokens = queryGPT(prompt_text, LLM, api_key, temperature, return_tokens=return_tokens)
            # print(reply, '\nAttempts:', attempts, '\n')

            try_query = False
        except openai.OpenAIError as e:
            print("ERROR! I'm going to sleep for " + str(attempts) + "s")
            print("Error:", str(e))

            time.sleep(attempts+2)
    if return_tokens:
        return reply, message, prompt_tokens, completion_tokens
    return reply, message

def isBadReply(reply, k):
    if len(reply) < k:
        return True
    else:
        return False


def LoadLLMRepliesFromTextFiles(output_dir):
    # get all .txt file names in folder that in the format _summary.txt
    file_names = [f for f in os.listdir(output_dir) if f.endswith('_summary.txt')]
    # sort the file names by the number in the file name
    file_names = sorted(file_names, key=lambda x: int(x.split('_')[0]))
    #(file_names)

    samples = []
    for file_name in file_names:
        file = open(output_dir + file_name, 'r')
        samples.append(file.read().split('REPLY:')[-1])
        file.close()

    return samples

# Made using GPT4
# def parse_comma_separated(input_str):
#     # Pattern for comma-separated format
#     pattern = r'\b([A-Z])\s*,'
#
#     matches = re.findall(pattern, input_str)
#     if matches:
#         # The last character in the sequence might not be followed by a comma, so we'll add it manually
#         last_char = re.search(r'([A-Z])\s*$', input_str)
#         if last_char:
#             matches.append(last_char.group(1))
#         return ''.join(matches)
#     return None

# Made using GPT4
# def parse_features_integrated(input_str):
#     # Check comma-separated format first
#     comma_result = parse_comma_separated(input_str)
#     if comma_result:
#         return comma_result
#
#     # Patterns for other formats
#     patterns = [
#         r'\bFeature\s([A-Z])\b',           # For format "Feature A"
#         r'\b\d[\).]\s*([A-Z])\b',          # For formats "1) A" and "1. A"
#         r'\b[a-zA-Z][\).]\s*([A-Z])\b'     # For format "a) A"
#     ]
#
#     # Search for the patterns in the input string
#     for pattern in patterns:
#         matches = re.findall(pattern, input_str)
#         if matches:
#             return ''.join(matches)
#
#     return None  # Return None if no patterns matched
#
#
def parse_extended_comma_format_final(input_str):
    pattern = r'(?:[a-zA-Z][\).]\s*[0-9A-Z]+\s*\n\nb\)|ending order:|features with the largest changes are|are:)\s*([A-Z](?:\s*,\s*[A-Z])*)\s*$'
    match = re.search(pattern, input_str)
    if match:
        return ''.join(match.group(1).replace(" ", "").split(','))
    return None

def parse_b_block_format(input_str):
    # Pattern to match a block format like:
    # are:
    # b) A, B, C, D
    pattern = r'b\)\s*([A-Z](?:\s*,\s*[A-Z])*)'
    match = re.search(pattern, input_str)
    if match:
        return ''.join(match.group(1).replace(" ", "").split(','))
    return None

def parse_comma_separated_format_refined(input_str):
    pattern = r'\b([A-Z])\s*,'
    matches = re.findall(pattern, input_str)
    if matches:
        # The last character in the sequence might not be followed by a comma, so we'll add it manually
        last_char = re.search(r'([A-Z])\s*$', input_str)
        if last_char:
            matches.append(last_char.group(1))
        return ''.join(matches)
    return None
def parse_feature_format(input_str):
    pattern = r'\bFeature\s([A-Z])\b'
    matches = re.findall(pattern, input_str)
    if matches:
        return ''.join(matches)
    return None

def parse_numbered_format(input_str):
    pattern = r'\b\d[\).]\s*([A-Z])\b'
    matches = re.findall(pattern, input_str)
    if matches:
        return ''.join(matches)
    return None

def parse_lettered_format(input_str):
    pattern = r'\b[a-zA-Z][\).]\s*([A-Z])\b'
    matches = re.findall(pattern, input_str)
    if matches:
        return ''.join(matches)
    return None

def parse_extended_comma_format_final(input_str):
    pattern = r'(?:[a-zA-Z][\).]\s*[0-9A-Z]+\s*\n\nb\)|ending order:|features with the largest changes are|are:)\s*([A-Z](?:\s*,\s*[A-Z])*)\s*$'
    match = re.search(pattern, input_str)
    if match:
        return ''.join(match.group(1).replace(" ", "").split(','))
    return None

def parse_input_final_refined_v3(input_str):
    # Reordering the parsers with the final version of the extended comma format parser
    parsers = [
        parse_b_block_format,
        parse_extended_comma_format_final,
        parse_feature_format,
        parse_numbered_format,
        parse_lettered_format,
        parse_comma_separated_format_refined
    ]

    for parser in parsers:
        result = parser(input_str)
        if result:
            return result

    return None  # Return None if no patterns matched


def parser_3_3(sample, LLM_top_k):
    rank = sample.rstrip('\n').split('\n')
    if 'might be' in rank[-1]:
        separator = 'might be'
    elif ' is' in rank[-1]:
        separator = ' is'
    elif ' are likely' in rank[-1]:
        separator = ' are likely'
    elif ' are' in rank[-1]:
        separator = ' are'
    else:
        separator = ':'

    remove_characters = ['and', '>', '=', ' ', '.', ',', ':', "'", '(', ')', '-', '\"', '*', '`']
    if '>' in rank[-1] or '=' in rank[-1]:
        text = rank[-1].split(separator)[-1].strip()
        for remove_character in remove_characters:
            text = text.replace(remove_character, '')
    elif rank[-1].split('ank:')[-1].startswith('This') or rank[-1].split('ank:')[-1].startswith('In'):
        j = -2
        while not rank[j]:
            j -= 1
        text = rank[j].split('ank:')[-1].strip()
        for remove_character in remove_characters:
            text = text.replace(remove_character, '')
    else:
        text = rank[-1].split(separator)[-1].strip().rstrip('.').lstrip()
        for remove_character in remove_characters:
            text = text.replace(remove_character, '')
        text = ''.join([t.replace(' ', '').replace('.', '') for t in text.split(', ')])
    # split the string into a list of characters
    text = list(text)
    return text[:LLM_top_k]

def parse_list_format(text):
    # Regular expression to match list items, considering various formats (numbered, bulleted, etc.)
    # This pattern now captures a broader set of characters after the marker
    pattern = re.compile(r'^\s*(?:\d+\.\s*|\*\s*|\-\s*)([A-Za-z0-9 ]+)(?=\s|$)', re.MULTILINE)
    
    # Find all matches in the text
    matches = pattern.findall(text)
    
    # Process matches to extract the last word or character if it's like 'Feature A', 'Feature B', etc.
    features = [match.split()[-1] for match in matches]
    
    return features


def parseLLMTopKsFromTxtFiles(samples, LLM_top_k, experiment_section='3.1'):
    print("Parsing LLM replies...")
    LLM_topks = []
    if experiment_section == 'llama':
        for s, sample in enumerate(samples):
            LLM_topks.append(parse_list_format(sample))
    elif experiment_section == '3.2':
        preds = []
        for s, sample in enumerate(samples):
            pred = sample.strip('\n').split('\n')[0]
            preds.append(pred)
            LLM_topks.append(parser_3_3(sample, LLM_top_k))
        LLM_topks = (preds, LLM_topks)
    elif experiment_section == '3.4':
        for s, sample in enumerate(samples):
            sample = sample.lstrip('\n').split('\n')[0]
            LLM_topks.append(parser_3_3(sample, LLM_top_k))
    elif experiment_section == '3.3':
        for s, sample in enumerate(samples):
            LLM_topks.append(parser_3_3(sample, LLM_top_k))
    else:
        for s, sample in enumerate(samples):
            sample = sample.rstrip('\n')
            topk = sample.split('\n')[-1]
            topk = topk.replace('.', '')

            if LLM_top_k == 1:
                if ':' in topk:
                    topk = topk.split(':')[1]
                topk = topk.replace('\'', '')
                topk = topk.replace('\"', '')
                topk = topk.replace(' ', '')
                topk = [topk]
            elif LLM_top_k > 1:
                print(topk)
                if len(topk) == 1:
                    # convert string into list of characters
                    topk = sample.split('\n')[-LLM_top_k:]
                    topk = str(topk)
                    topk = topk[1:-1]
                    # convert array of strings to string.
                if ':' in topk:
                    topk = topk.split(':')[1]
                topk = topk.replace('\'', '')
                topk = topk.replace('\"', '')
                topk = topk.replace(' ', '')
                topk = topk.split(',')
                #Remove any element from the list that's not a letter in the alphabet
                print(topk)
                topk = [item for item in topk if item.isalpha()]
                # if the topk is a single string of letters, split it into a list of letters
                print(topk)

            LLM_topks.append(topk)
    return LLM_topks


def removeBadReplies(LLM_topks, orig_inds, k):
    # loop each answer from the LLM and if it gave a list of
    # anything less than the requested amount of features, k, remove it (for now)
    num_bad_replies = 0
    LLM_topks_temp = copy.deepcopy(LLM_topks)
    orig_inds = list(orig_inds)
    for l, LLM_topk in reversed(list(enumerate(LLM_topks_temp))):
        isAllLetters = all(len(item) == 1 and item.isalpha() and item.isupper() and (item in string.ascii_uppercase[:k]) for item in LLM_topk)

        # if reply is e.g. ["'D'", "'E'", "'F'", "'A'", "'B'"], convert it to ['D', 'E', 'F', 'A', 'B']
        if not isAllLetters:
            LLM_topk          = [item.replace("'", "") for item in LLM_topk]
            LLM_topk          = [item.replace("\"", "") for item in LLM_topk]
            LLM_topks_temp[l] = LLM_topk
            isAllLetters      = all(len(item) == 1 and item.isalpha() and item.isupper() and (item in string.ascii_uppercase) for item in LLM_topk)

        if len(LLM_topk) < k or not isAllLetters: # or len(LLM_topk) > k 
            #print("Bad reply:", LLM_topk, "at loop index", l, "and orig index", orig_inds[l])
            del LLM_topks_temp[l]
            del orig_inds[l]
            num_bad_replies += 1
        elif len(LLM_topk) > k:
            LLM_topks_temp[l] = LLM_topk[:k]
    print("Found", num_bad_replies, "bad replies")

    return LLM_topks_temp, orig_inds

def get_response_processor(LLM_name):
    if LLM_name == "GPT-4":
        return GPT4ResponseProcessor()
    elif LLM_name in LLM_APIS.keys():
        return APIResponseProcessor(LLM_name)
    else:
        raise ValueError("Invalid LLM name")

class ResponseProcessor:
    """
    Shared base class for processing responses.
    Shared functionality might include:
    - Error handling/logging (example shown for GPT4)
    - Extracting top-k feature integers from a top-k feature string
    """
    def __init__(self, LLM_name):
        self.LLM_name = LLM_name
        self.logger = logging.getLogger(__name__)

    def handle_error(self, error):
        """Handle error during response processing."""
        self.logger.error(f"Error processing response for {self.LLM_name}: {error}")

    def log_processing_details(self, details):
        """Log some details about the processing."""
        self.logger.info(details)

    def process_response(self, response):
        raise NotImplementedError("Subclasses must implement this method")


class GPT4ResponseProcessor(ResponseProcessor):
    def __init__(self):
        super().__init__("GPT-4")

    def process_response(self, response):
        try:
            pass
            # Specific implementation for GPT-4
            # ... processing code ...
        except Exception as error:
            self.handle_error(error)
        finally:
            self.log_processing_details("GPT-4 processing completed")

class APIResponseProcessor(ResponseProcessor):
    def __init__(self, LLM_name):
        super().__init__(LLM_name)

    def process_top_k(self, response, question):
        if self.LLM_name == "bloom-176b":
            # Bloom repeats the prompt, so we need to remove it
            response = response.split(question)[1]
            
            # Remove anything after the first period
            response = response.split('.')[0]

            # Replace "and" with a comma
            response = response.replace(' and', ',')
            
            # Remove any trailing punctuation
            response = [r.strip() for r in response.split(',') if r.strip() != '']
            return response
        else:
            return response.split(question)


# Older functions for processing predictions (may or may not come in useful)

def process_float_value(value_str):
    try:
        value = float(value_str)
        return value
    except ValueError:
        return None

def process_answer(answer):
    """
    Process answer from LLM
    answer: str, answer from LLM
    """
    # Check if the answer is a valid float value and return it
    float_value = process_float_value(answer)
    if float_value is not None:
        return float_value
    
    float_value = process_float_value(answer.split(' ')[0])
    if float_value is not None:
        return float_value

    # Split the answer string using "Answer: " as the separator
    split_answer = answer.split("Answer: ")

    # If there is a valid split, check the last part of the split for the answer
    if len(split_answer) > 1:
        # Iterate through the characters in the extracted answer and attempt to parse the float value
        for delimiter in [' ', '\n', ',']:
            extracted_answer = split_answer[-1].split(delimiter)[0]

            # Check if the extracted_answer is a valid float value and return it
            float_value = process_float_value(extracted_answer)
            if float_value is not None:
                return float_value

    # If no valid answer is found, return None or raise an exception
    else:
        return None
        # Or, alternatively, raise an exception:
        # raise ValueError("Invalid answer format")
