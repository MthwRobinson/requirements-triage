# Script for verifying that run_bart_sum can be invoked from its directory
export OUTPUT_DIR_NAME=bart_ghub
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
python finetune.py \
--data_dir=gh_issues/ \
--model_name_or_path=t5-small \
--learning_rate=3e-5 \
--train_batch_size=2 \
--eval_batch_size=2 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=1  \
--gpus=0 \
--do_train "$@" \
--overwrite_output_dir
