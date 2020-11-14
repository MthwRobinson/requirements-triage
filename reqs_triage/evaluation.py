import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def performance_metrics(data):
    """Measures performance based on model outputs

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing the model outputs

    Returns
    -------
    performance : pd.DataFrame
        A dataframe containing the performance metrics for each model
    """
    performance = {"model": [], "metric_type": [], "score": []}
    for metric_type, metric_function in METRICS.items():
        for column in data:
            if _check_column(column):
                score = metric_function(data["actual"], data[column])
                performance["score"].append(score)
                performance["model"].append(column)
                performance["metric_type"].append(metric_type)
    return pd.DataFrame(performance)


def _check_column(column):
    """Checks the columns because we only want to run performance metrics on the columns
    that contain model outputs.

    Parameters
    ----------
    column : str
        The name of the column

    Returns
    -------
    good_column : bool
        True if it is a column that we want to evaluate
    """
    if column == "t5":
        return True
    elif column.endswith("tfidf") or column.endswith("bow"):
        return True
    else:
        return False
