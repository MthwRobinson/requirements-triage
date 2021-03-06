# Script for verifying that run_bart_sum can be invoked from its directory
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=/home/matt/models/feature-request/bart_multilabel_large_xfer_25

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
python finetune.py \
--data_dir=/home/matt/data/ghub_labels_xfer_small \
--model_name_or_path=/home/matt/models/feature-request/bart_multilabel_large/best_tfmr \
--learning_rate=3e-5 \
--train_batch_size=2 \
--eval_batch_size=2 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=4  \
--val_metric=loss \
--gpus=0 \
--do_train "$@" \
