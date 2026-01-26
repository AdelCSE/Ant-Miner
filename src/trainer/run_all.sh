#!/bin/bash

# 1. Define Datasets
datasets=(
    "yeast3" "abalone19" "segment0" "pageblocks"
)

# 2. Define Fitness Combinations (each entry is "obj1 obj2")
fitness_groups=(
    "specificity sensitivity"
)

# 3. Nested Loop
for d in "${datasets[@]}"; do
    for f in "${fitness_groups[@]}"; do

        echo "--------------------------------------------"
        echo "Dataset: $d"
        echo "Objectives: $f"
        echo "--------------------------------------------"

        #python3 trainer.py --dataset "$d" --objs $f 
        python3 moea2_trainer.py --dataset "$d" --objs $f
        python3 moea2_trainer.py --dataset "$d" --objs $f --archive-type rulesets --rulesets subproblem

    done
done

echo ""
echo "All experiments completed."
read -p "Press Enter to exit"
