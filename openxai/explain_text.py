
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_text import LimeTextExplainer

# Path to the All_Beauty reviews file you downloaded
reviews_file_path = '../data/Beauty/All_Beauty.jsonl'

# Load the reviews and the ratings
texts, ratings = [], []
with open(reviews_file_path, 'r') as file:
    for line in file:
        review = json.loads(line.strip())
        texts.append(review['text'])
        ratings.append(review['rating'])

# get indices where number of words is > 5 and number of words is < 100
indices = [i for i, text in enumerate(texts) if 5 < len(text.split()) < 100]
texts = [texts[i] for i in indices]
ratings = [ratings[i] for i in indices]

# Convert ratings to binary sentiment (1 for positive, 0 for negative)
sentiments = [1 if rating >= 4 else 0 for rating in ratings]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, sentiments, test_size=0.2, random_state=0)

# Create a TfidfVectorizer and Logistic Regression pipeline
pipeline = make_pipeline(TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), stop_words='english'),
                         LogisticRegression(solver='liblinear', random_state=0))

# get num features from vectorizer
# vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), stop_words='english')
# num_features = vectorizer.fit_transform(X_train).shape[1]
# print('num feats', num_features)
num_features_to_explain = 4

# Train the classifier on the training data
pipeline.fit(X_train, y_train)

# Predict sentiments on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Use LIME to explain a single prediction from the test set
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

# Choose a random instance to explain
# idx = np.random.randint(0, len(X_test))
idx = 60435
exp, neighborhood_sentences = explainer.explain_instance(X_test[idx], pipeline.predict_proba, num_features=num_features_to_explain,
                                                         return_neighborhood_sentences=True)


def generate_ordered_sentence_neighborhood(sentence, num_samples=5000):
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




## select neighborhood samples for a prompt!
num_icl = 16

# get the class label for the neighborhood instances
neighborhood_preds = np.argmax(pipeline.predict_proba(neighborhood_sentences), 1)

# calculate number of negatives and positives in the neighborhood
num_negatives, num_positives = np.sum(neighborhood_preds == 0), np.sum(neighborhood_preds == 1)

# confirm that there are at least 16/2 negative and 16/2 positive instances in the neighborhood
while num_negatives < num_icl // 2 or num_positives < num_icl // 2:
    print("Not enough instances in the neighborhood for a balanced ICL prompt for index %d" % idx)
    print("You have {} negative instances and {} positive instances".format(num_negatives, num_positives))
    # get a new instance to explain
    idx = np.random.randint(0, len(X_test))
    exp, neighborhood_sentences = explainer.explain_instance(X_test[idx], pipeline.predict_proba, num_features=num_features_to_explain,
                                                             return_neighborhood_sentences=True)
    neighborhood_preds = np.argmax(pipeline.predict_proba(neighborhood_sentences), 1)

    # calculate number of negatives and positives in the neighborhood
    num_negatives, num_positives = np.sum(neighborhood_preds == 0), np.sum(neighborhood_preds == 1)

print('There are {} negatives and {} positives'.format(num_negatives, num_positives))

# first sentence is the original sentence! We just want the neighborhood sentences
neighborhood_sentences = neighborhood_sentences[1:]
neighborhood_preds = neighborhood_preds[1:]

original_sentence = X_test[idx]
original_prediction = pipeline.predict([original_sentence])[0]
# class balancing - find indices for num_icl // 2 instances for each class
negatives_idx = np.where(neighborhood_preds == 0)[0]
positives_idx = np.where(neighborhood_preds == 1)[0]

# get the negative and positive instances
negatives = [neighborhood_sentences[i] for i in negatives_idx]
positives = [neighborhood_sentences[i] for i in positives_idx]

# get the neighborhood predictions for the negatives
negatives_preds = pipeline.predict(negatives)
positives_preds = pipeline.predict(positives)

print('Original sentence:', original_sentence)
print('Original prediction:', original_prediction)
dataset_prompt = ''
for s, sentence in enumerate(range(num_icl // 2)):
    dataset_prompt += 'Kept words: ' + negatives[s] + '\n' + 'Change in output: ' + str(original_prediction - negatives_preds[s]) + '\n\n'
    dataset_prompt += 'Kept words: ' + positives[s] + '\n' + 'Change in output: ' + str(original_prediction - positives_preds[s]) + '\n\n'

print(dataset_prompt)



# Now make a prompt with the removed words
print('Original prediction:', original_prediction)
dataset_prompt = ''
for s, sentence in enumerate(range(num_icl // 2)):
    neg_sent = negatives[s].split()
    pos_sent = positives[s].split()
    # take set difference of original sentence (all words) with the neighborhood sentence (kept words)
    neg_removed = ' '.join(list(set(original_sentence.split()) - set(neg_sent)))
    pos_removed = ' '.join(list(set(original_sentence.split()) - set(pos_sent)))

    dataset_prompt += 'Removed words: ' + neg_removed + '\n' + 'Change in output: ' + str(original_prediction - negatives_preds[s]) + '\n\n'
    dataset_prompt += 'Removed words: ' + pos_removed + '\n' + 'Change in output: ' + str(original_prediction - positives_preds[s]) + '\n\n'

print(dataset_prompt)
print('done')
print('done')

# find where in x_test the sentence 'Too expensive for such s small amount.' is
for i, x in enumerate(X_test):
    if x == 'Too expensive for such s small amount.':
        print(i)
        break

# # Show the explanation for the predicted class
# print('Document id: %d' % idx)
# print('Predicted class =', pipeline.classes_[exp.predicted_class], '\nProbability(score) =', exp.predict_proba[exp.predicted_class])
# print('True class: %s' % ('Positive' if y_test[idx] == 1 else 'Negative'))
# print('\nExplanation for prediction:')
# print('\n'.join(map(str, exp.as_list())))
# # exp.save_to_file(f'lime_explanation_{idx}.html')

# End of script
