#!/usr/bin/env bash
#SBATCH --job-name=amazontitle131k
#SBATCH --output=output.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=128
#SBATCH --partition=cscc-cpu-p
#SBATCH --time=20:00:00
#SBATCH --qos=cscc-cpu-qos

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
cd "$SCRIPT_DIRECTORY"

# Experiment configuration
readonly MODEL_PATH="/nfs-stor/linh.vu/runs/amazon131k_zs_tree_20260312011956/linear_pipeline.pickle"
readonly DATA_ROOT="/nfs-stor/linh.vu/LF-AmazonTitle-131K/lf/zs"
readonly DATASET_NAME="LF-AmazonTitles-131K"
readonly STRATEGY="rank_rrf"  # raw, rank_rrf, or rank_normal
readonly RUN_NAME="amazontitles131k_zs_${STRATEGY}"

# Input files
readonly TRAIN_INSTANCE_DATA_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_train_30.svm"
readonly TEST_INSTANCE_DATA_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_test_30.svm"
readonly LABEL_FEATURE_PATH="${DATA_ROOT}/${DATASET_NAME}_tfidf_lf_30.svm"

case "$STRATEGY" in
    raw | rank_rrf | rank_normal) ;;
    *)
        echo "Invalid strategy: $STRATEGY" >&2
        exit 1
        ;;
esac

for input_file in \
    "$MODEL_PATH" \
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
echo "Strategy: $STRATEGY"

python3 model_predict.py \
    --model_path "$MODEL_PATH" \
    --train_instance_data_path "$TRAIN_INSTANCE_DATA_PATH" \
    --test_instance_data_path "$TEST_INSTANCE_DATA_PATH" \
    --label_feature_path "$LABEL_FEATURE_PATH" \
    --run_name "$RUN_NAME" \
    --strategy "$STRATEGY"
