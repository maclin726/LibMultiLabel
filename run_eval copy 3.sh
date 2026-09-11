# python3 main.py --data_name MIMIC\
#     --training_file data/MIMIC/MIMIC_tfidf_train_ext.svm \
#     --test_file data/MIMIC/MIMIC_tfidf_test_ext.svm \
#     --data_format svm \
#     --eval \
#     --checkpoint_path /home/lvu5/LibMultiLabel/runs/MIMIC_ovr_sklearn_20250709121429/linear_pipeline.pickle \
#     --linear 
    
    # --linear_technique 1vsrest \
    # --model_name ovr_20250624115103 \
# runs/eurlex57k_sian_june_9th_logistic_ovr_20250609152227/linear_pipeline.pickle


python3 main.py --data_name AmazonCat-13K\
    --training_file /home/lvu5/LibMultiLabel/data/AmazonCat-13K/amazoncat13k_tfidf_train_ext.svm \
    --test_file /home/lvu5/LibMultiLabel/data/AmazonCat-13K/amazoncat13k_tfidf_test_ext.svm \
    --data_format svm \
    --eval \
    --checkpoint_path /home/lvu5/LibMultiLabel/runs/AmazonCat-13K_ovr_20251111144536/linear_pipeline.pickle \
    --linear