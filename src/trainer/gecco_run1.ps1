# 1. Define Datasets
$datasets = @("tictactoe", "mushrooms", "hepatitis", "ljubljana", "chess", "cargood") #("flare", "zoo3", "yeast3", "segment0", "pageblocks", "abalone19")

# 2. Define Fitness Combinations
$f = @("confidence", "simplicity")

# 3. Define archive types (2 variants)
$archive_types = @("rules", "rulesets") 

# 4. Define configurations (population + ant)
$configs = @(
    @{ pop = 300;  ant = 1 },
    @{ pop = 50; ant = 6 },
    @{ pop = 30; ant = 10 },
    @{ pop = 10; ant = 30 },
    @{ pop = 6; ant = 50 }
)

# 5. Nested loops
foreach ($d in $datasets) {
    foreach ($archive in $archive_types) {
        foreach ($cfg in $configs) {

            Write-Host "--------------------------------------------" -ForegroundColor Cyan
            Write-Host "Dataset: $d"
            Write-Host "Objectives: $f"
            Write-Host "Archive Type: $archive"
            Write-Host "Subproblems: $($cfg.pop)"
            Write-Host "Ants: $($cfg.ant)"
            Write-Host "--------------------------------------------" -ForegroundColor Cyan
            $args = @(
                "--dataset", $d,
                "--objs", $f,
                "--archive-type", $archive,
                "--population", $cfg.pop,
                "--ant", $cfg.ant
            )
                
            if ($archive -ne "rules") {
                $args += @("--rulesets", "subproblem")
            }
            
            python moea2_trainer.py @args
        }
    }
}

Write-Host "`nAll experiments completed." -ForegroundColor Green
Read-Host -Prompt "Press Enter to exit"