import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline
import tqdm


DATA_DIR = "/home/matt/data/ghub_labels"
MODEL_DIR = "/home/matt/models/feature-request"


def train_model(classifier, vectorizer, data):
    """Trains a feature request identification classifier using the specified classifier
    and vectorizer.

    Parameters
    ----------
    classifier : skelarn.Classifier
        An sklearn style classifier
    vectorizer : skelarn.Vectorizer
        An sklearn style vectorizer
    data : pd.DataFrame
        A dataframe with the source requirement text and label

    Returns
    -------
    classifier : skelarn.Classifier
        An sklearn style classifier
    vectorizer : skelarn.Vectorizer
        An sklearn style vectorizer
    """
    X = vectorizer.fit_transform(data["source"])
    y = np.array(data["target"] == "feature").astype(int)
    classifier.fit(X, y)
    return classifier, vectorizer


def t5_inference(data):
    """Performs inference on the dataset using the trained T5 model

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe with the source requirement text and label

    Returns
    -------
    data : pd.DataFrame
        A dataframe with the source requirement text and label and the T5 inference
        added
    """
    model_path = os.path.join(MODEL_DIR, "t5_multilabel_base", "best_tfmr")
    summarizer = pipeline("summarization", model=model_path)

    t5_inference = []
    for i in tqdm.tqdm(data.index):
        source = data.loc[i]["source"]
        result = summarizer(source, max_length=2)
        t5_inference.append(result[0]["summary_text"])

    data["t5"] = t5_inference
    return data
