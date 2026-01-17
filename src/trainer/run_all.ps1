# 1. Define Datasets
$datasets = @("mushrooms", "tictactoe", "hepatitis", "ljubljana", "cargood", "chess", "zoo3", "flare", "yeast3", "abalone19", "segment0", "pageblocks")

# 2. Define Fitness Combinations
$fitness_groups = @(
    @("specificity", "sensitivity"),
    @("confidence", "simplicity"),
    @("confidence", "specificity"),
    @("confidence", "sensitivity")
)

# 3. Nested Loop
foreach ($d in $datasets) {
    foreach ($f in $fitness_groups) {
        
        Write-Host "--------------------------------------------" -ForegroundColor Cyan
        Write-Host "Dataset: $d"
        # $f is an array, passing it like this prints it cleanly
        Write-Host "Objectives: $f"
        Write-Host "--------------------------------------------" -ForegroundColor Cyan
        
        # PowerShell automatically unpacks the array $f into separate arguments
        # So this executes: python trainer.py --dataset yeast3 --objs confidence simplicity
        
        python moea_trainer.py --dataset $d --objs $f --archive-type rulesets --rulesets iteration
    }
}

Write-Host "`nAll experiments completed." -ForegroundColor Green
Read-Host -Prompt "Press Enter to exit"