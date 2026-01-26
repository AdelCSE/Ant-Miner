# 1. Define Datasets
$datasets = @("mushrooms", "tictactoe", "hepatitis", "ljubljana", "cargood", "chess") # , "zoo3", "flare", "yeast3", "abalone19", "segment0", "pageblocks"

# 2. Define Fitness Combinations
$f= @("specificity", "sensitivity")

# 3. Nested Loop
foreach ($d in $datasets) {
        
        Write-Host "--------------------------------------------" -ForegroundColor Cyan
        Write-Host "Dataset: $d"
        Write-Host "Objectives: $f"
        Write-Host "--------------------------------------------" -ForegroundColor Cyan
        
        #python trainer.py --dataset $d --objs $f
        python moea2_trainer.py --dataset $d --objs $f --archive-type rulesets --rulesets iteration
}

Write-Host "`nAll experiments completed." -ForegroundColor Green
Read-Host -Prompt "Press Enter to exit"