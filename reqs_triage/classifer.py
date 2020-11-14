import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from transformers import pipeline
import tqdm

from reqs_triage.dataset import read_dataset


DATA_DIR = "/home/matt/data/ghub_labels"
MODEL_DIR = "/home/matt/models/feature-request"


MODELS = [
    ("rf-tfidf", RandomForestClassifier(), TfidfVectorizer()),
    ("rf-bow", RandomForestClassifier(), CountVectorizer()),
    ("lr-tfidf", LogisticRegression(), TfidfVectorizer()),
    ("lf-bow", LogisticRegression(), CountVectorizer()),
    ("nb-tfidf", BernoulliNB(), TfidfVectorizer()),
    ("nb-bow", BernoulliNB(), CountVectorizer()),
    ("knn-tfidf", KNeighborsClassifier(), TfidfVectorizer()),
    ("knn-bow", KNeighborsClassifier(), CountVectorizer()),
    ("svm-tfidf", SVC(), TfidfVectorizer()),
    ("svm-bow", SVC(), CountVectorizer()),
]


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


def train_and_save_models():
    """Trains all of the models in the list and serializes them to disk."""
    train = read_dataset(DATA_DIR, "train")
    for label, classifier, vectorizer in tqdm.tqdm(MODELS):
        classifier, vectorizer = train_model(classifier, vectorizer, train)

        with open(os.path.join(MODEL_DIR, f"{label}-classifier.pkl"), "wb") as f:
            pickle.dump(classifier, f)

        with open(os.path.join(MODEL_DIR, f"{label}-vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)


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
