'''
This file is for evaluating the quality of our RAG system
using the Hairy Trumpet tool/dataset.
'''

import sys
import ragnews
import csv
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class RAGEvaluator:
    def predict(self, masked_text):
        ''' 
        >>> model = RAGEvaluator()
        >>> model.predict('There no mask token here.')
        []
        >>> model.predict('[MASK0] is the democratic nominee')
        ['Harris']
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        '''
        # you might think about:
        # calling the ragnews.run_llm function directly;
        # so we will call the ragnews.rag function

        valid_labels = ['Harris', 'Trump']

        db = ragnews.ArticleDB('ragnews.db')
        textprompt = f'''
            This is a fancier question that is based on standard cloze style benchmarks.
            I'm going to provide you a sentence, and that sentence will have a masked token inside of it that will look like [MASK0] or [MASK1] or [MASK2] and so on.
            And your job is to tell me what the value of that masked token was.
            Valid values include: {valid_labels}

            The size of you output should just be a single word for each mask.
            You should not provide any explanation or other extraneous words.
            If there are multiple mask tokens, provide each token separately with a whitespace in between.

            INPUT: [MASK0] is the democratic nominee
            OUTPUT: Harris

            INPUT: [MASK0] is the democratic nominee and [MASK1] is the republican nominee
            OUTPUT: Harris Trump

            INPUT: {masked_text}
            OUTPUT: '''
        output = ragnews.rag(textprompt, db, keywords_text=masked_text)
        return output

        # Reasons why bad results:
        # 1. the code (esp the prompt) in this function is bad
        # 2. the rag function itself could be bad
        #   
        # In order to improve (1) above:
        # 1. prompt engineering didn't work
        # 2. I had to change what model I was using

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    )

class RAGClassifier:
    def __init__(self, valid_labels=None):
        """
        Initializes the classifier with a list of valid labels.

        Parameters:
        valid_labels (list): List of valid labels that the classifier can predict.
        """
        self.valid_labels = valid_labels if valid_labels is not None else []

    def predict(self, X):
        """
        Predicts the masked tokens based on the input text using the RAG system.

        Parameters:
        X (list of str): A list of masked text inputs.
        
        Returns:
        list: A list of predicted labels for each input.
        """
        db = ragnews.ArticleDB('ragnews.db')
        predictions = []

        for masked_text in X:
            textprompt = f'''
            Please predict the value of the masked tokens.
            Valid values include: {self.valid_labels}

            INPUT: {masked_text}
            OUTPUT: '''
            try:
                output = ragnews.rag(textprompt, db, keywords_text=masked_text)
            except AssertionError:
                output = "No articles found"
            predictions.append(output)
        
        return predictions
    
logging.basicConfig(level=logging.INFO)
def load_data(filename):
    """
    Load the HairyTrumpet dataset from a JSON file.
    Each row in the file is expected to have a sentence with masked tokens and the correct labels.

    Returns:
    X (list of str): The list of input masked sentences.
    y (list of str): The list of correct labels.
    valid_labels (set): The set of all valid labels present in the dataset.
    """
    X = []
    y = []
    valid_labels = set()

    with open(filename, 'r') as f:
        for line in f:
            dp = json.loads(line)  # Parse the JSON line
            masked_sentence = dp['masked_text']
            labels = dp['masks']
            
            X.append(masked_sentence)
            y.append(labels)
            valid_labels.update(labels)  # Collect unique labels

    logging.info(f'Valid labels: {valid_labels}')

    return X, y, list(valid_labels)


def main(filename):
    # Load data from the HairyTrumpet file
    X, y, valid_labels = load_data(filename)

    # Initialize the classifier with the valid labels
    classifier = RAGClassifier(valid_labels=valid_labels)

    # Run predictions on all instances
    y_pred = classifier.predict(X)

    #testing
    print(f"Length of X: {len(X)}")
    print(f"Length of y: {len(y)}")
    print(f"Length of y_pred: {len(y_pred)}")

    # Flatten the list of true labels (y) and predicted labels (y_pred) to compute accuracy
    y_true_flat = [label for labels in y for label in labels]
    y_pred_flat = [pred for preds in y_pred for pred in preds.split()]

    #testing
    print(f"y_true_flat length: {len(y_true_flat)}")
    print(f"y_pred_flat length: {len(y_pred_flat)}")

    # Compute accuracy
    accuracy = accuracy_score(y_true_flat, y_pred_flat)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py ./hairy-trumpet")
        sys.exit(1)
    
    # The command-line argument is the data file path
    data_file_path = sys.argv[1]
    main(data_file_path)