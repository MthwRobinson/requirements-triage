import re

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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
    issue_text = re.sub(CODE_BLOCK_RE, "", issue_text, 0, re.DOTALL)
    issue_text = issue_text.replace("\r", " ").replace("\n", " ")
    return issue_text


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
    label = label.lower()
    return NORMALIZED_LABELS.get(label, None)


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
        WHERE LENGTH(body) > 1000
        AND array_length(labels, 1) > 0
        AND {label_where_clause}
        ORDER BY RANDOM()
        LIMIT {num_examples}
    """
    data = pd.read_sql(sql, connection)
    data["body"] = data["body"].apply(lambda body: remove_code_blocks(body))
    data["text"] = data["title"] + " " + data["body"]
    data["labels"] = data["labels"].apply(lambda labels: labels_to_text(labels))
    return data


def prepare_dataset(directory, num_examples=500):
    data = get_examples(num_examples=num_examples)
    data = shuffle(data)
    data.reset_index(drop=True, inplace=True)

    test_start = int(0.7 * len(data))
    val_start = int(0.9 * len(data))

    csv_kwargs = {"index": False, "header": False, "line_terminator": "\n"}

    data[:test_start]["text"].to_csv(f"{directory}/train.source", **csv_kwargs)
    data[:test_start]["labels"].to_csv(f"{directory}/train.target", **csv_kwargs)

    data[test_start:val_start]["text"].to_csv(f"{directory}/test.source", **csv_kwargs)
    data[test_start:val_start]["labels"].to_csv(
        f"{directory}/test.target", **csv_kwargs
    )

    data[val_start:]["text"].to_csv(f"{directory}/val.source", **csv_kwargs)
    data[val_start:]["labels"].to_csv(f"{directory}/val.target", **csv_kwargs)
