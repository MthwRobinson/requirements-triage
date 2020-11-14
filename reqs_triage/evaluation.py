import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def bootstrap_evaluation(data, sample_size=250, iterations=100):
    """Evaluates the model outputs on a random subsample of the test data.

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing the model outputs
    sample_size : int
        The samples size for the evaluation run
    iterations : int
        The number of iterations to run

    Returns
    -------
    performance : pd.DataFrame
        A dataframe containing the performance metrics for each model and an id column
        for each bootstrap iteration
    """
    performance = []
    eval_data = data.copy()
    for iteration in range(iterations):
        eval_data = eval_data.sample(frac=1).reset_index(drop=True)
        iter_performance = performance_metrics(eval_data[:sample_size])
        iter_performance["iteration"] = iteration
        performance.append(iter_performance)
    return pd.concat(performance)


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
