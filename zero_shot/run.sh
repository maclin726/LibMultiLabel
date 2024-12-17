#!/bin/bash

# Change the data root and model path accordingly
model_path="../runs/eurlex57k_linear_tree_default_params_20241207110126/linear_pipeline.pickle"
data_root="../dataset/EURLEX57K/from_hezhe/v1"

train_instance_data_path="$data_root/eurlex57k_tfidf_train_ext.svm"
test_instance_data_path="$data_root/eurlex57k_tfidf_test_ext.svm"
label_data_path="$data_root/eurlex57k_tfidf_lf.svm"

task(){
# Set up train command
cmd="python model_predict.py"
cmd="${cmd} --model_path $model_path"
cmd="${cmd} --train_instance_data_path $train_instance_data_path"
cmd="${cmd} --test_instance_data_path $test_instance_data_path"
cmd="${cmd} --label_data_path $label_data_path"
echo $cmd

}

# Check command
task
wait

# Run
multiprocess_num=1
task | xargs -0 -d '\n' -P $multiprocess_num -I {} sh -c {}
