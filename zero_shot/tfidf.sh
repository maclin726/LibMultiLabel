#!/usr/bin/env bash
#SBATCH --job-name=amazon131k_tfidf
#SBATCH --output=output.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=245G
#SBATCH --cpus-per-task=128
#SBATCH --partition=cscc-cpu-p
#SBATCH --time=48:00:00
#SBATCH --qos=cscc-cpu-qos

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
cd "$SCRIPT_DIRECTORY"

# Experiment configuration
readonly MODEL_PATH="unused"
readonly DATA_ROOT="/nfs-stor/linh.vu/LF-Amazon-131K/zs"
readonly DATASET_NAME="LF-Amazon-131K"
readonly RUN_NAME="amazon131k_legacy_zs_tfidf_raw"

# Input files
readonly TRAIN_INSTANCE_DATA_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_train_ext.svm"
readonly TEST_INSTANCE_DATA_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_test_ext.svm"
readonly LABEL_FEATURE_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_lf.svm"

for input_file in \
    "$TRAIN_INSTANCE_DATA_PATH" \
    "$TEST_INSTANCE_DATA_PATH" \
    "$LABEL_FEATURE_PATH"; do
    if [[ ! -f "$input_file" ]]; then
        echo "Missing input file: $input_file" >&2
        exit 1
    fi
done

hostname
echo "Run name: $RUN_NAME"

python3 model_tfidf_predict.py \
    --model_path "$MODEL_PATH" \
    --train_instance_data_path "$TRAIN_INSTANCE_DATA_PATH" \
    --test_instance_data_path "$TEST_INSTANCE_DATA_PATH" \
    --label_feature_path "$LABEL_FEATURE_PATH" \
    --run_name "$RUN_NAME"
