import os
import random
import re

import numpy as np
import pandas as pd

import reqs_triage.database as db

CODE_BLOCK_RE = r"(```).*?(```)"

# The source labels are all labels that appeared at least 400 times in the dataset
NORMALIZED_LABELS = {
    "bug": "bug",
    "enhancement": "feature",
    "question": "question",
    "feature": "feature",
    "documentation": "documentation",
    "help wanted": "support",
    "feature request": "feature",
    "dependencies": "dependencies",
    "discussion": "discussion",
    "improvement": "feature",
    "support": "support",
    "type: feature": "feature",
    "feature-request": "feature",
    "type: question": "support",
    "type: support": "support",
    "suggestion": "feature",
    "type:bug": "bug",
    "new feature": "feature",
    "type=enhancement": "feature",
    ":bug: bug": "bug",
    "type:bug/performance": "bug",
    "bug report": "bug",
    "area: bug :bug:": "bug",
}


ADDITIONAL_LABELS = {
    "performance": "performance",
    "windows": "os",
    "ios": "os",
    "android": "os"
}


def remove_code_blocks(issue_text):
    if not isinstance(issue_text, str):
        return ""
    issue_text = re.sub(CODE_BLOCK_RE, "", issue_text, 0, re.DOTALL)
    issue_text = issue_text.replace("\r", " ").replace("\n", " ")
    return issue_text


@np.vectorize
def build_text(title, body):
    title = title.replace("\n", " ")
    body = body[:1000].replace("\n", " ")
    text = f"{title} --- {body}"
    if len(text.split()) < 10:
        return ""
    else:
        return f"multilabel classification: {text}"


def labels_to_text(labels):
    normalized_labels = set()
    for label in labels:
        normalized_label = _normalize_label(label)
        if normalized_label:
            normalized_labels.add(normalized_label)

    if normalized_labels:
        return " , ".join(list(normalized_labels))
    else:
        return ""


def _normalize_label(label):
    """Lowercases the label and converts it to a standardized form

    Parameters
    ----------
    label : str
        The original label

    Returns
    -------
    label : str
        The normalized label
    """
    return NORMALIZED_LABELS.get(label.lower(), None)


def _build_label_where_clause():
    labels = list(NORMALIZED_LABELS.keys())
    where = " OR ".join([f"'{label}' ILIKE ANY(labels)\n" for label in labels])
    return f"( {where} )"


def get_examples(num_examples=500):
    connection = db.connect()
    label_where_clause = _build_label_where_clause()
    sql = f"""
        SELECT id, package_id, title, body, labels
        FROM open_source.issues
        WHERE {label_where_clause}
        AND title IS NOT NULL
        AND length(TITLE) > 20
        AND array_length(labels, 1) = 1
        ORDER BY RANDOM()
        LIMIT {num_examples}
    """
    data = pd.read_sql(sql, connection)
    data["body"] = data["body"].apply(lambda body: remove_code_blocks(body))
    data["text"] = build_text(data["title"], data["body"])
    data["labels"] = data["labels"].apply(lambda labels: labels_to_text(labels))
    return data


def train_test_split(package_ids):
    """Splits the package ids into the training and tests sets, keeping all of the
    issues from the same package together.

    Parameters
    ----------
    package_ids : list
        A list of package ids

    Returns
    -------
    train_ids : list
        The package ids in the training set
    test_ids : list
        The package ids in the test set
    val_ids : list
        The package ids in the validation set
    """
    random.shuffle(package_ids)

    test_start = int(0.8 * len(package_ids))
    val_start = int(0.9 * len(package_ids))

    train_ids = package_ids[:test_start]
    test_ids = package_ids[test_start:val_start]
    val_ids = package_ids[val_start:]

    return train_ids, test_ids, val_ids


def save_dataset(data, directory, filename):
    """Saves the data in a format that can be processed by the transformers training
    script.

    Parameters
    ----------
    data : pd.DataFrame
        The data set to save
    directory : str
        The target directory
    filename : str
        The file prefix
    """
    print(f"Dataset size for {filename}: {len(data)}")
    csv_kwargs = {"index": False, "header": False, "line_terminator": "\n"}

    data["text"].to_csv(f"{directory}/{filename}.source", **csv_kwargs)
    data["labels"].to_csv(f"{directory}/{filename}.target", **csv_kwargs)

    with open(f"{directory}/{filename}.packages", "w") as f:
        for package_id in data["package_id"].unique():
            f.write(f"{package_id}\n")


def prepare_dataset(directory, num_examples=500):
    """Queries the postgres database, normalizes the labels, and save the dataset in a
    format that can be processed by the transformers training script.

    Parameters
    ----------
    directory : str
        The directory to save the output
    num_examples : int
        The number of examples to pull from the database
    """
    data = get_examples(num_examples=num_examples)
    data = data[data["labels"] != ""]
    data = data[data["text"] != ""]

    package_ids = list(data["package_id"].unique())
    train_ids, test_ids, val_ids = train_test_split(package_ids)

    train = data[data["package_id"].isin(train_ids)]
    save_dataset(train, directory, "train")

    test = data[data["package_id"].isin(test_ids)]
    save_dataset(test, directory, "test")

    val = data[data["package_id"].isin(val_ids)]
    save_dataset(val, directory, "val")


def read_dataset(directory, prefix):
    """Reads the data set from

    Parameters
    ----------
    directory : str
        The target directory
    prefix : str
        The file prefix

    Returns
    -------
    data : pd.DataFrame
        A dataframe with the source text for the requirements and the label
    """
    source_filename = os.path.join(directory, f"{prefix}.source")
    with open(source_filename, "r") as f:
        source = f.read().split("\n")

    target_filename = os.path.join(directory, f"{prefix}.target")
    with open(target_filename, "r") as f:
        target = f.read().split("\n")

    return pd.DataFrame({"source": source, "target": target})
