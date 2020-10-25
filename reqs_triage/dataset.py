import re

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import reqs_triage.database as db

CODE_BLOCK_RE = r"(```).*?(```)"


def remove_code_blocks(issue_text):
    issue_text = re.sub(CODE_BLOCK_RE, "", issue_text, 0, re.DOTALL)
    issue_text = issue_text.replace("\r", " ").replace("\n", " ")
    return issue_text


def labels_to_text(labels):
    if labels:
        return "; ".join(labels)
    else:
        return ""


def get_examples(num_examples=500):
    connection = db.connect()
    sql = f"""
        SELECT body, labels
        FROM open_source.issues
        WHERE LENGTH(body) > 1000
        AND array_length(labels, 1) > 0
        ORDER BY RANDOM()
        LIMIT {num_examples}
    """
    data = pd.read_sql(sql, connection)
    data["body"] = data["body"].apply(lambda body: remove_code_blocks(body))
    data["labels"] = data["labels"].apply(lambda labels: labels_to_text(labels))
    return data


def prepare_dataset(directory, num_examples=500):
    data = get_examples(num_examples=num_examples)
    data = shuffle(data)
    data.reset_index(drop=True, inplace=True)

    test_start = int(0.7 * len(data))
    val_start = int(0.9 * len(data))

    csv_kwargs = {"index": False, "header": False, "line_terminator": "\n"}

    data[:test_start]["body"].to_csv(f"{directory}/train.source", **csv_kwargs)
    data[:test_start]["labels"].to_csv(f"{directory}/train.target", **csv_kwargs)

    data[test_start:val_start]["body"].to_csv(f"{directory}/test.source", **csv_kwargs)
    data[test_start:val_start]["labels"].to_csv(
        f"{directory}/test.target", **csv_kwargs
    )

    data[val_start:]["body"].to_csv(f"{directory}/val.source", **csv_kwargs)
    data[val_start:]["labels"].to_csv(f"{directory}/val.target", **csv_kwargs)
