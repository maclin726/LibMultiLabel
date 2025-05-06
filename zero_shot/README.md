# Run Experiments


```
./run.sh
```
Note: Please change `run.sh` accordingly before running; check the script for more details.

Define your strategy before running an experiment. There are currently 3 strategies:
    - raw
    - rank_rrf
    - rank_normal 
raw: Only use the raw scores.
rank_rrf: Reciprocal Rank Fusion.
rank_normal: Ranking only.

run_name: to specify the name of the log file.

```
#!/bin/bash

# Change the data root and model path accordingly
# full dataset
model_path="../runs/eurlex57k_sian_ovr_traintest_20250428090728/linear_pipeline.pickle"
data_root="../data/eurlex57k_sian"
run_name="eurlex57k_ovr_rrf_sian_may5"
# raw, rank_rrf, rank_normal
strategy="raw"

# toy dataset
# model_path="../runs/toy_eurlex57k_tree_20250210125133/linear_pipeline.pickle"
# data_root="../data/toy_eurlex57k"

train_instance_data_path="$data_root/eurlex57k_tfidf_train_ext.svm"
test_instance_data_path="$data_root/eurlex57k_tfidf_test_ext.svm"
label_feature_path="$data_root/eurlex57k_tfidf_lf.svm"

task(){

# Set up train command
cmd="python3 model_predict_reorg.py"
cmd="${cmd} --model_path $model_path"
cmd="${cmd} --train_instance_data_path $train_instance_data_path"
cmd="${cmd} --test_instance_data_path $test_instance_data_path"
cmd="${cmd} --label_feature_path $label_feature_path"
cmd="${cmd} --run_name $run_name"
cmd="${cmd} --strategy $strategy"
echo $cmd

}

# Check command
task
wait

# Run
multiprocess_num=1
task | xargs -0 -d '\n' -P $multiprocess_num -I {} sh -c {}

```

