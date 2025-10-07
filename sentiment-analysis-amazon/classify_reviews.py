import nltk
import logging
import argparse
import pandas as pd

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_json(file_path, lines=True)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        logger.info(f"Head of the dataset:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by removing unnecessary columns and handling missing values.
    
    Args:
        df (pd.DataFrame): Input dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping preprocessing.")
        return df
    
    columns_to_keep = ['reviewText', 'overall', 'reviewTime']

    df = df[columns_to_keep]
    
    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip().lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [WordNetLemmatizer().lemmatize(word, wordnet.VERB) for word in tokens if word not in stop_words]
        return ' '.join(tokens).strip()

    df['reviewTextClean'] = df['reviewText'].apply(preprocess_text)
    return df

def classify_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify the sentiment of reviews using a pre-trained transformer model.
    
    Args:
        df (pd.DataFrame): Input dataset with preprocessed review texts.
    Returns:
        pd.DataFrame: Dataset with an additional column for sentiment labels.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping sentiment classification.")
        return df
    
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def get_sentiment(text: str) -> str:
        if not text:
            return "neutral"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        return predicted_class_id

    df['sentiment'] = df['reviewTextClean'].apply(get_sentiment)
    df = df[['reviewTextClean', 'overall', 'reviewTime', 'sentiment']]
    return df

def write_output_to_drive():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis for Amazon Reviews")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    args = parser.parse_args()

    df = load_data(args.input_file)
    df = preprocess_data(df)
    df = classify_sentiments(df)
    write_output_to_drive(df)