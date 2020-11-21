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


def transformer_inference(data, model_name="bart", size="large", xfer=False):
    """Performs inference on the dataset using the trained transformer model

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
    model_postfix = f"multilabel_{size}" if not xfer else f"multilabel_{size}_xfer"
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{model_postfix}", "best_tfmr")
    summarizer = pipeline("summarization", model=model_path)

    t5_inference = []
    for i in tqdm.tqdm(data.index):
        source = data.loc[i]["source"]
        result = summarizer(source, max_length=6)
        t5_inference.append(result[0]["summary_text"].strip())

    data[f"{model_name}_label"] = t5_inference
    target_label = "performance" if xfer else "feature"
    data[model_name] = (data[f"{model_name}_label"] == target_label).astype(int)
    return data


def _model_inference(label, classifier, vectorizer, data):
    """Performs inference on the dataset using each of the trained sklearn models

    Parameters
    ----------
    label : str
        The label for the model, which is used in the column name
    classifier : skelarn.Classifier
        An sklearn style classifier
    vectorizer : skelarn.Vectorizer
        An sklearn style vectorizer
    data : pd.DataFrame
        A dataframe with the source requirement text and label

    Returns
    -------
    data : pd.DataFrame
        A dataframe with the source requirement text and label and the T5 inference
        added
    """
    X = vectorizer.transform(data["source"])
    data[label] = classifier.predict(X)
    return data


def inference(data):
    """Performs inference on the dataset using each of the trained sklearn models

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
    for label, _, _ in tqdm.tqdm(MODELS):
        with open(os.path.join(MODEL_DIR, f"{label}-classifier.pkl"), "rb") as f:
            classifier = pickle.load(f)

        with open(os.path.join(MODEL_DIR, f"{label}-vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        data = _model_inference(label, classifier, vectorizer, data)
    return data
