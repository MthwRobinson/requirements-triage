# Script for verifying that run_bart_sum can be invoked from its directory
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=/home/matt/models/t5_ghub1106_20201107

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
python finetune.py \
--data_dir=/home/matt/data/ghub_labels_20201107 \
--model_name_or_path=t5-small \
--learning_rate=3e-5 \
--train_batch_size=10 \
--eval_batch_size=10 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=1  \
--max_target_length=1 \
--gpus=0 \
--do_train "$@" \
--overwrite_output_dir
