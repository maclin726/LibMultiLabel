#!/usr/bin/env bash
#SBATCH --job-name=amazon131k_zs_tfidf
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
readonly DATA_ROOT="/nfs-stor/linh.vu/LF-Amazon-131K-ZS/LF-Amazon-131K/zeroshot"
readonly DATASET_NAME="LF-Amazon-131K"
readonly RUN_NAME="amazon131k_zs_tfidf_raw"
readonly PROPENSITY_A="0.6"
readonly PROPENSITY_B="2.6"

# Input files
readonly TRAIN_INSTANCE_DATA_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_train.svm"
readonly TEST_INSTANCE_DATA_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_test.svm"
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
echo "PSP propensity: A=$PROPENSITY_A, B=$PROPENSITY_B"

python3 model_tfidf_predict.py \
    --model_path "$MODEL_PATH" \
    --train_instance_data_path "$TRAIN_INSTANCE_DATA_PATH" \
    --test_instance_data_path "$TEST_INSTANCE_DATA_PATH" \
    --label_feature_path "$LABEL_FEATURE_PATH" \
    --run_name "$RUN_NAME" \
    --propensity_a "$PROPENSITY_A" \
    --propensity_b "$PROPENSITY_B"
