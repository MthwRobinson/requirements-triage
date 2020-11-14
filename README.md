## Requirements Triage

This is code developed in support of requirements triage for CrowdRE.

### Installation

To install the package, you can run `make pip-install` from the root directory.

### Environmental Variables

To prepare the data set, you need to be connected to a Postgres database. The
`database.py` file manages the database connection, and expects to find DB connection
string information in the following environmental variables

```
PG_HOST
PG_PORT
PG_DB
PG_USER
```

Database credentials must be stored in a `.pgpass` file.

## Dataset Preparation

Once you have installed the package, you can prepare a dataset for training in the
target directory of your choice by running

```python
from reqs_triage.dataset import prepare_dataset

directory = "/my/cool/directory"
prepare_dataset(directory, num_examples=100000)

```

## Model Training

### T5 Model

Model training for the core model for this experiment uses the Huggingface
`transformers` package, which is included as one of the dependencies. Once you have
prepared the training data, you can train the fine-tuned T5 model by running

```
sh training/finetune_t5.sh
```

Due to the size of the model, it cannot be stored in the GitHub repository. Once the
model is finalized, we will provide a link to a Huggingface repository where the models
weights can be downloaded and used.

### Sklearn Models

In addition to the T5 model, we test the performance of a number of classicial ML models
that are mentioned in the literature. To run the model training for these models, you
can run the following code:

```python
from reqs_triage.classifier import train_and_save_models()

train_and_save_models
```

## Outputs

The model predictions for each of the models can be found in the `output` directory. THe
directory contains the following files:

1. `results.csv` - Contains the test examples, the actual label, and the model
   predictions for each of the models.
1. `performance.csv` - Measures the performance of the model on the full data set with
   respect to a number of a performance metrics
1. `bootstrap_evaluation.py` - Measures performance on a 100 random subsamples of the
   test dataset to measure variance in performance
