label_tree_dir="label_tree_obj"

dataset="txt eurlex"
# dataset="svm amazoncat-13k"
# dataset="svm wiki10-31k"
# dataset="svm wiki-500k"
# dataset="svm amazon-670k"
# dataset="svm amazon-3m"

for clus in elkan
do  
    for dataset in "txt eurlex" "svm amazoncat-13k" "svm wiki10-31k"
    do
        for seed in {0..4}
        do
            echo "python3 tree_space.py build_tree $dataset --cluster $clus --K 100 --dmax 2 --seed $seed"
            time python3 tree_space.py build_tree $dataset $label_tree_dir \
                --cluster $clus \
                --K 100 \
                --dmax 2 \
                --seed $seed
        done
    done
done
