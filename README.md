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

Model training for the core model for this experiment uses the Huggingface
`transformers` package, which is included as one of the dependencies. Once you have
prepared the training data, you can train the fine-tuned T5 model by running

```
sh training/finetune_t5.sh
```
