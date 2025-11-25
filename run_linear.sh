# python3 main.py --data_name MIMIC_sklearn\
#     --model_name ovr \
#     --training_file data/MIMIC/full_grow/MIMIC_tfidf_train_ext.svm \
#     --test_file data/MIMIC/full_grow/MIMIC_tfidf_test_ext.svm \
#     --linear --linear_technique 1vsrest \
#     --data_format svm

# python3 main.py --data_name eurlex57k_sian_zslwan_svm_4197\
#     --model_name ovr \
#     --training_file data/eurlex57k_sian_zslwan/eurlex57k_tfidf_train_ext.svm \
#     --test_file data/eurlex57k_sian_zslwan/eurlex57k_tfidf_test_ext.svm \
#     --linear --linear_technique 1vsrest \
#     --data_format svm

# python3 main.py --data_name eurlex4k\
#     --model_name ovr \
#     --training_file data/eurlex4k/eurlex4k_tfidf_train_ext_FULLtest.svm \
#     --test_file data/eurlex4k/eurlex4k_tfidf_test_ext_FULLtest.svm \
#     --linear --linear_technique 1vsrest \
#     --data_format svm

python3 main.py --data_name AmazonCat-13K\
    --model_name ovr \
    --training_file data/AmazonCat-13K/amazoncat13k_tfidf_train_ext.svm \
    --test_file data/AmazonCat-13K/amazoncat13k_tfidf_test_ext.svm \
    --linear --linear_technique tree \
    --data_format svm
