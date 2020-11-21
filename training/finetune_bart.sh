# Script for verifying that run_bart_sum can be invoked from its directory
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=/home/matt/models/bart_multilabel_large

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
python finetune.py \
--data_dir=/home/matt/data/ghub_labels \
--model_name_or_path=facebook/bart-large-cnn \
--learning_rate=3e-5 \
--train_batch_size=4 \
--eval_batch_size=4 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=1  \
--val_metric=loss \
--gpus=0 \
--do_train "$@" \
