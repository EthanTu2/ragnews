import sys
import ragnews
import json
import logging
import re
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Set up logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

class RAGClassifier:
    def __init__(self, valid_labels=None, db_path='ragnews.db'):
        """
        Initializes the classifier with a list of valid labels and a preloaded ArticleDB instance.
        """
        self.valid_labels = valid_labels if valid_labels is not None else []
        self.db = ragnews.ArticleDB(db_path)  # Load DB once for reuse

    def predict(self, X):
        """
        Predicts the masked tokens based on the input text using the RAG system.
        """
        predictions = []
        
        # Adjust the pattern to match the masked token format like [MASK0], [MASK1], etc.
        mask_pattern = r"\[MASK\d+\]"  # This pattern matches [MASK0], [MASK1], etc.

        for masked_text in X:
            # Find all unique masked tokens in the text using a set
            unique_masks = set(re.findall(mask_pattern, masked_text))
            n = len(unique_masks)

            textprompt = f'''
            You are a statistician tasked with predicting masked tokens.
            Do not provide any explanations or descriptions.
            For each masked token, provide a one word prediction in string format.
            Your response must contain exactly {n} predictions, one for each masked token, and nothing else.
            The valid prediction values are: {self.valid_labels}. 
            If a prediction is not among the valid prediction values, do not include it in the output.
            INPUT: {masked_text}
            OUTPUT: '''

            try:
                output = ragnews.rag(textprompt, self.db, keywords_text=masked_text)
            except AssertionError:
                output = "article not found"

            # Split the output by whitespace or commas, and filter out empty entries
            individual_predictions = re.split(r'[,\s\n]+', output.strip())

            # Ensure that we only append the first n predictions
            print("\n\n\nnumber of tokens = ", n, "\n\n\n", "predicted tokens = ", individual_predictions[:n], "\n\n\n")

            # Ensure that predictions correctly match one of the valid labels
            for j in range(len(individual_predictions[:n])):
                match = False
                for label in self.valid_labels:
                    if individual_predictions[j] == label:
                        match = True
                # If no matches are found, truncate the individual prediction so it matches a valid label
                if (not match):
                        i = 0
                        for letter in individual_predictions[j]:
                            i +=1
                            if individual_predictions[j][:i] in self.valid_labels:
                                break
                        individual_predictions[j] = individual_predictions[j][:i]
            
            predictions.extend(individual_predictions[:n])  # Append each prediction individually
        
        return predictions


def clean_prediction(pred):
    """
    Cleans the prediction string by removing unwanted characters and splitting multiple predictions into individual ones.
    """
    # Step 1: Remove unwanted characters like square brackets or quotes
    cleaned_pred = re.sub(r"[\[\]']", '', pred)
    
    # Step 2: Split the predictions by common delimiters (comma, space, or newline)
    predictions = re.split(r'[,\n\s]+', cleaned_pred.strip())
    
    # Step 3: Filter out empty predictions
    relevant_predictions = [p.strip() for p in predictions if p.strip()]
    
    # Return the list of cleaned, individual predictions
    return relevant_predictions


def load_data(filename):
    """
    Load the dataset from a JSON file.
    """
    X, y = [], []
    valid_labels = set()

    with open(filename, 'r') as f:
        for line in f:
            try:
                dp = json.loads(line)
                X.append(dp['masked_text'])
                y.append(dp['masks'])
                valid_labels.update(dp['masks'])
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON line: {e}")

    logging.info(f'Valid labels: {valid_labels}')
    return X, y, list(valid_labels)


def main(filename):
    # Load data
    X, y, valid_labels = load_data(filename)

    # Initialize the classifier
    classifier = RAGClassifier(valid_labels=valid_labels)

    # Make predictions
    y_pred = classifier.predict(X)

    # Testing
    print("y_pred length =", len(y_pred))
    print("y_pred =", y_pred)

    # Flatten lists for accuracy computation
    y_true_flat = [label for labels in y for label in labels]
    
    # Clean and flatten the predicted output
    y_pred_flat = [pred for y in y_pred for pred in clean_prediction(y)]

    print("y_true_flat =", y_true_flat)
    print("y_pred_flat =", y_pred_flat)

    # Validate lengths match before calculating accuracy
    if len(y_true_flat) != len(y_pred_flat):
        logging.warning(f"Mismatch in lengths of true and predicted labels! y_true_flat length: {len(y_true_flat)}, y_pred_flat length: {len(y_pred_flat)}")
        return  # Stop if lengths don't match to avoid invalid accuracy calculation

    # Compute accuracy
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py ./hairy-trumpet")
        sys.exit(1)

    data_file_path = sys.argv[1]
    main(data_file_path)