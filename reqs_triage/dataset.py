import random
import re

import numpy as np
import pandas as pd

import reqs_triage.database as db

CODE_BLOCK_RE = r"(```).*?(```)"

# The source labels are all labels that appeared at least 400 times in the dataset
NORMALIZED_LABELS = {
    "bug": "bug",
    "enhancement": "feature request",
    "question": "question",
    "feature": "feature request",
    "documentation": "documentation",
    "help wanted": "support",
    "feature request": "feature request",
    "dependencies": "dependencies",
    "discussion": "discussion",
    "improvement": "feature request",
    "support": "support",
    "type: feature": "feature request",
    "feature-request": "feature request",
    "type: question": "support",
    "type: support": "support",
    "suggestion": "feature request",
    "type:bug": "bug",
    "new feature": "feature request",
    "type=enhancement": "feature request",
    ":bug: bug": "bug",
    "type:bug/performance": "bug",
    "bug report": "bug",
    "area: bug :bug:": "bug",
}


def remove_code_blocks(issue_text):
    if not isinstance(issue_text, str):
        return ""
    issue_text = re.sub(CODE_BLOCK_RE, "", issue_text, 0, re.DOTALL)
    issue_text = issue_text.replace("\r", " ").replace("\n", " ")
    return issue_text


@np.vectorize
def build_text(title, body):
    if not body:
        return title
    else:
        body = body[:1000]
        return f"feature request classification: {title} --- {body}"


def labels_to_text(labels):
    normalized_labels = []
    for label in labels:
        normalized_label = _normalize_label(label)
        if normalized_label:
            normalized_labels.append(normalized_label)

    if normalized_labels:
        return "; ".join(normalized_labels)
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
    label = NORMALIZED_LABELS.get(label, label)
    return str(int(label == "feature request"))


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
        WHERE array_length(labels, 1) = 1
        AND length(body) > 100
        AND title IS NOT NULL
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

    test_start = int(0.7 * len(package_ids))
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

    package_ids = list(data["package_id"].unique())
    train_ids, test_ids, val_ids = train_test_split(package_ids)

    train = data[data["package_id"].isin(train_ids)]
    save_dataset(train, directory, "train")

    test = data[data["package_id"].isin(test_ids)]
    save_dataset(test, directory, "test")

    val = data[data["package_id"].isin(val_ids)]
    save_dataset(val, directory, "val")
