import string
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class TextClassifierPrompt:
    def __init__(self, prompt_dict):
        self.pre_text = prompt_dict['pre_text']
        self.post_text = prompt_dict['post_text']

    # Generate embeddings
    def generate_embeddings(self, texts, embedding_model):

        batch_size = 256  # Adjust based on your system's memory capacity
        embeddings = []
        print('transforming data...')
        num_batches = len(texts) // batch_size
        for i in range(0, len(texts), batch_size):
            if i % 100 == 0:
                print(f'Processing batch {i // batch_size + 1}/{num_batches}')
            batch = texts[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    def create_prompt(self, original_sentence, embedding, classifier, num_ICL):
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # original prediction
        with torch.no_grad():
            original_prediction = np.argmax(classifier(torch.tensor(embedding[np.newaxis, :])), 1).item()

        neighborhood_sentences = self.generate_neighborhood_sentences(original_sentence, num_samples=500)

        # calculate embeddings for the neighborhood instances
        print('generating embeddings')
        neighborhood_embeddings = self.generate_embeddings(neighborhood_sentences, embedding_model)
        print('embeddings generated')

        # get the class label for the neighborhood instances
        with torch.no_grad():
            neighborhood_preds = np.argmax(classifier(torch.tensor(neighborhood_embeddings)), 1)

        # calculate number of negatives and positives in the neighborhood
        num_negatives, num_positives = torch.sum(neighborhood_preds == 0).item(), torch.sum(neighborhood_preds == 1).item()

        # confirm that there are at least 16/2 negative and 16/2 positive instances in the neighborhood
        if num_negatives < num_ICL // 2 or num_positives < num_ICL // 2:
            print("Not enough instances in the neighborhood for a balanced ICL prompt for the sentence:\n",
                  original_sentence)
            print("You have {} negative instances and {} positive instances".format(num_negatives, num_positives))

        # class balancing - find indices for num_icl // 2 instances for each class
        negatives_idx = np.where(neighborhood_preds == 0)[0]
        positives_idx = np.where(neighborhood_preds == 1)[0]


        # get the negative and positive instances
        negatives = [neighborhood_sentences[i] for i in negatives_idx]
        positives = [neighborhood_sentences[i] for i in positives_idx]

        print('Original sentence:', original_sentence)
        print('Original prediction:', original_prediction)
        dataset_prompt = ''
        loop_counter, neg_counter, pos_counter = 0, 0, 0

        show_kept_words = False

        while neg_counter + pos_counter < num_ICL:
            if show_kept_words:
                # alternate between negative and positive instances
                if loop_counter % 2 == 0 and neg_counter < num_negatives:
                    dataset_prompt += '\nKept words: ' + negatives[neg_counter] + '\n' + 'Change in output: ' + str(
                        original_prediction - 0) + '\n'
                    neg_counter += 1
                    loop_counter += 1
                elif pos_counter < num_positives:
                    dataset_prompt += '\nKept words: ' + positives[pos_counter] + '\n' + 'Change in output: ' + str(
                        original_prediction - 1) + '\n'
                    pos_counter += 1
                    loop_counter += 1
            else:
                # alternate between negative and positive instances
                if loop_counter % 2 == 0 and neg_counter < num_negatives:
                    neg_sent = negatives[neg_counter].split()
                    neg_removed = ' '.join(list(set(original_sentence.split()) - set(neg_sent)))
                    dataset_prompt += '\nRemoved words: ' + neg_removed + '\n' + 'Change in output: ' + str(
                        original_prediction - 0) + '\n'
                    neg_counter += 1
                    loop_counter += 1
                elif pos_counter < num_positives:
                    pos_sent = positives[pos_counter].split()
                    pos_removed = ' '.join(list(set(original_sentence.split()) - set(pos_sent)))
                    dataset_prompt += '\nRemoved words: ' + pos_removed + '\n' + 'Change in output: ' + str(
                        original_prediction - 1) + '\n'
                    pos_counter += 1
                    loop_counter += 1


        return self.pre_text + '\nOriginal sentence: ' + original_sentence + '\n' + dataset_prompt + self.post_text

    def generate_neighborhood_sentences(self, sentence, num_samples=5000):
        words = sentence.split()
        num_words = len(words)

        # Initialize the list to store neighborhood sentences
        neighborhood_sentences = []  # Include the original sentence as the first sample

        for _ in range(num_samples):
            num_words_to_remove = np.random.randint(1, num_words)  # Number of words to remove
            words_to_remove = np.random.choice(range(num_words), size=num_words_to_remove, replace=False)
            perturbed_sentence = ' '.join([word for idx, word in enumerate(words) if idx not in words_to_remove])
            neighborhood_sentences.append(perturbed_sentence)

        return neighborhood_sentences



class Prompt:
    def __init__(self, feature_names, input_str='Input:', output_str='Prediction: ',
                 input_sep='\n\n', output_sep='. ', feature_sep='\n', value_sep='=', n_round=5,
                 hide_feature_details=False, hide_test_sample=False,
                 hide_last_pred=False, conversion=None, suffixes=None, feature_types=None,
                 use_soft_preds=False, add_explanation=False, prompt_id='chirag_v1'):
        """
        Store data for prompt creation
        feature_names: list of str, feature names

        Optional:
        input_str: str, default 'Input: ', string to prepend to input.
                   If input_str == 'sample', then display the sample number next to each sample.
        output_str: str or list, default 'Prediction: ', string(s) to prepend to output
                    if a list is passed, each element corresponds to each class label.
        input_sep: str, default '\n\n', separator between inputs (x,y pairs)
        output_sep: str, default '. ', separator between x and y in each pair
        feature_sep: str, default '\n', separator between features
        value_sep: str, default '=', separator between feature name and value
        n_round: int, default 5, number of decimal places to round to
        hide_feature_details: bool, default False, if True, feature names are hidden and "A" "B" "C" are used instead
        hide_test_sample: bool, default False, if True, test sample in prompt is hidden
        conversion: list of functions, default None, feature value to string conversion
        suffixes: list of str, default None, suffixes to add to feature values
        add_explanation: flag for adding explanations in ICL prompt
        prompt_id: id of the prompting style
        """
        self.feature_names = feature_names
        self.input_str = input_str
        self.output_str = output_str
        self.input_sep = input_sep
        self.output_sep = output_sep
        self.feature_sep = feature_sep
        self.value_sep = value_sep
        self.n_round = n_round
        self.hide_feature_details = hide_feature_details
        self.feature_types = feature_types
        self.use_soft_preds = use_soft_preds
        self.add_explanation = add_explanation
        self.prompt_id = prompt_id
        self.hide_test_sample = hide_test_sample
        self.hide_last_pred = hide_last_pred

        # Conversion functions (feature values -> text e.g. 0 -> 'Male')
        self._null_conversion = []
        for i in range(len(feature_names)):
            if self.feature_types[i] == 'c':
                self._null_conversion.append(lambda x: "{:.{}f}".format(x, self.n_round))
            elif self.feature_types[i] == 'd':
                self._null_conversion.append(lambda x: str(int(x)))
        self.conversion = self._null_conversion if conversion is None else conversion

        # Suffixes (e.g. ' Months' for COMPAS length of stay)
        self._null_suffixes = ['']*len(feature_names)
        self.suffixes = self._null_suffixes if suffixes is None else suffixes

    def single(self, x, y=None):
        """
        Create text for (x,y) pair, given index
        x: array-like, single input point
        y: int/float, model prediction (label)
        If y is None, no model prediction is added
        """
        if self.hide_feature_details:
            # f'Feature {string.ascii_uppercase[i]}' for longer names
            feature_names        = [string.ascii_uppercase[i] for i in range(len(self.feature_names))]
            conversion, suffixes = self._null_conversion, self._null_suffixes
        else:
            feature_names        = self.feature_names
            conversion, suffixes = self._null_conversion, self._null_suffixes
        x_text = [str(feature) + self.value_sep + conversion[i](val) + str(suffixes[i])
                  for i, (feature, val) in enumerate(zip(feature_names, x))]

        if y is not None:
            if isinstance(self.output_str, list):
                y_text = self.output_str[int(y)]  # not compatible with soft preds
            elif isinstance(self.output_str, str):
                pred_str = "{:.{}f}".format(y, self.n_round) if self.use_soft_preds else str(int(y))
                y_text = self.output_str + pred_str
            else:
                raise ValueError('output_str must be str or list')
        else:
            y_text = ''
        return self.feature_sep.join(x_text) + self.output_sep + y_text


    def multiple(self, X, y=None):
        """
        Create text for multiple (x,y) pairs
        X: array-like, input data
        y: array-like, output data
        If y is None, no model predictions are added.
        """
        if y is not None:
            assert len(X) == len(y)

        text = ''
        for i, x in enumerate(X):
            if i == len(X) - 1 and self.hide_last_pred:
                text += self.input_str + self.single(x, y=None) + self.output_sep + self.output_str + '\n'
                last_y = y[i] if y is not None else None
            else:
                yi = y[i] if y is not None else None
                text += self.input_str + self.single(x, y=yi) + self.input_sep
        if self.hide_last_pred:
            return text, last_y
        return text
 
    def icl_explanation(self, X, E, Y):
        """
        X: test samples for generating explanations from post hoc explainers
        E: post hoc explanations
        """
        text = ''
        for x, e, y in zip(X, E, Y):
            text += '\nInput: ' + self.single(x, y).split(';')[0] + '\nExplanation: ' + ','.join([string.ascii_uppercase[i] for i in np.argsort(np.abs(e))[::-1]]) + '\n'
        return text
       
    def create_prompt(self, X_train, y_train, x, post_text, x_test,
                      explanations, y_test, y=None, pre_text='', mid_text=''):
        """
        Create prompt text for a given input
        X_train: array-like, training data
        y_train: array-like, training labels
        x: array-like, single test input
        question: str, question to be asked
        """

        if self.add_explanation:
            train_text = self.icl_explanation(x_test, explanations, y_test)
            test_text = 'Input: ' + self.single(x, y=y) + '\nExplanation: '
        elif self.hide_test_sample:
            if self.hide_last_pred:
                train_text, last_y = self.multiple(X_train, y_train)
            else:
                train_text = self.multiple(X_train, y_train)
            test_text  = ''
        else:
            train_text = self.multiple(X_train, y_train)
            test_text = self.single(x, y=y)

        if self.input_str.lower() == 'sample':
            if self.add_explanation:
                output = pre_text + train_text + 'Input: ' + test_text.split(';')[0] + '\n' + test_text.split(';')[1].strip()
            else:
                if self.hide_test_sample:
                     output = pre_text + '\n' + train_text + post_text             
                else:
                    output = pre_text + '\n' + train_text + mid_text + '\n' + test_text + '\n' + post_text
        else:
            if self.hide_test_sample:
                train_text = train_text
                output = pre_text + train_text + mid_text + post_text  # MAIN EXPS
            else:
                output = pre_text + train_text.lstrip('\n') + mid_text + test_text + post_text

        if self.hide_last_pred:
            return output, last_y
        return output
