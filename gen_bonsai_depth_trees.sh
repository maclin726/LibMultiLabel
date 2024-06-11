depths=(1 2 3 4 5 6)

# dataset="txt eurlex"
# fanouts=(64 64 16 8 6 4)

# dataset="svm amazoncat-13k"
# fanouts=(116 116 24 11 7 5)

# dataset="svm wiki10-31k"
# fanouts=(176 176 32 14 8 6)

# dataset="svm amazon-670k"
# fanouts=(819 819 88 29 15 10)

# dataset="svm amazon-3m"
# fanouts=(1677 1677 142 41 20 12)

dataset="svm wiki-500k"
fanouts=(708 708 80 27 14 9)

for clus in elkan
do  
    for i in "${!depths[@]}"
    do
        for seed in {0..4}
        do
            echo "python3 tree_space.py build_tree $dataset --cluster $clus --K ${fanouts[i]} --dmax ${depths[i]} --seed $seed"
            time python3 tree_space.py build_tree $dataset label_tree_obj \
             --cluster $clus \
             --K "${fanouts[i]}" \
             --dmax "${depths[i]}" \
             --seed $seed
        done
    done
done
