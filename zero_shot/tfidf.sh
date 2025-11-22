#!/bin/bash
# runs/MIMIC_ovr_sklearn_20250709121429
# /home/lvu5/LibMultiLabel/data/MIMIC

model_path="../runs/eurlex57k_sian_zslwan_svm_4197_ovr_full_label_space/linear_pipeline.pickle"
data_root="../data/eurlex4k"
run_name="eurlex57k_ovr_tfidf_raw"

# train_instance_data_path="$data_root/eurlex57k_tfidf_train_ext_FULL.svm"
# test_instance_data_path="$data_root/eurlex57k_tfidf_test_ext_FULL.svm"
# label_feature_path="$data_root/eurlex57k_tfidf_lf_FULL.svm"

# train_instance_data_path="$data_root/eurlex4k_tfidf_train_ext_FULL1.svm"
# test_instance_data_path="$data_root/eurlex4k_tfidf_test_ext_FULL1.svm"
# label_feature_path="$data_root/eurlex4k_tfidf_lf_FULL1.svm"

train_instance_data_path="$data_root/eurlex4k_tfidf_train_ext_FULL1.svm"
test_instance_data_path="$data_root/eurlex4k_tfidf_test_ext_FULL1.svm"
label_feature_path="$data_root/eurlex4k_tfidf_lf_FULL1.svm"

# Change the data root and model path accordingly
# full dataset
# model_path="../runs/MIMIC_sklearn_ovr_full/linear_pipeline.pickle"
# data_root="../data/MIMIC"
# run_name="MIMIC_ovr_tfidf_raw"

# train_instance_data_path="$data_root/full_set/MIMIC_tfidf_train_ext.svm"
# test_instance_data_path="$data_root/full_set/MIMIC_tfidf_test_ext.svm"
# label_feature_path="$data_root/full_set/MIMIC_tfidf_lf.svm"

task(){
# Set up train command
cmd="python3 tfidf_raw.py"
cmd="${cmd} --model_path $model_path"
cmd="${cmd} --train_instance_data_path $train_instance_data_path"
cmd="${cmd} --test_instance_data_path $test_instance_data_path"
cmd="${cmd} --label_feature_path $label_feature_path"
cmd="${cmd} --run_name $run_name"

echo $cmd

}

# Check command
task
wait

# Run
multiprocess_num=1
task | xargs -0 -d '\n' -P $multiprocess_num -I {} sh -c {}
