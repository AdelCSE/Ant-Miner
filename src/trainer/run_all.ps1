# 1. Define Datasets
$datasets = @("segment0") #, "zoo3", "flare", "yeast3", "abalone19", "segment0", "pageblocks"

# 2. Define Fitness Combinations
$f= @("specificity", "sensitivity")


# 3. Nested Loop
foreach ($d in $datasets) {
        
        Write-Host "--------------------------------------------" -ForegroundColor Cyan
        Write-Host "Dataset: $d"
        # $f is an array, passing it like this prints it cleanly
        Write-Host "Objectives: $f"
        Write-Host "--------------------------------------------" -ForegroundColor Cyan
        
        # PowerShell automatically unpacks the array $f into separate arguments
        # So this executes: python trainer.py --dataset yeast3 --objs confidence simplicity
        
        python trainer.py --dataset $d --objs $f
        #python moea_trainer.py --dataset $d --objs $f
        #python moea_trainer.py --dataset $d --objs $f --archive-type rulesets --rulesets iteration
}

Write-Host "`nAll experiments completed." -ForegroundColor Green
Read-Host -Prompt "Press Enter to exit"