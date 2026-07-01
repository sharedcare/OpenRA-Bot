param(
    [string]$BinDir = "F:/Projects/OpenRA/bin",
    [int]$Updates = 100,
    [int]$NumSteps = 128,
    [int]$MaxEpisodeTicks = 1800,
    [int]$WarmstartEpisodes = 3,
    [int]$WarmstartEpochs = 3,
    [double]$LearningRate = 7e-5,
    [double]$ClipCoef = 0.05,
    [double]$TargetKl = 0.03,
    [double]$EntCoef = 0.02,
    [int]$UpdateEpochs = 4,
    [int]$MinibatchSize = 64,
    [double]$GoalAlignedWeight = 0.6,
    [double]$TeacherKlCoef = 0.0,
    [int]$TeacherKlAnnealSteps = 50,
    [string]$RunName = "",
    [switch]$NoWarmstart,
    [switch]$LoadBcActionHead,
    [switch]$NoGoalConditioning,
    [switch]$AddOpponent
)

$ErrorActionPreference = "Stop"

$BotRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $BotRoot

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if ($NoGoalConditioning) {
        $goalTag = "no_goal"
    } else {
        $weightTag = [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:0.##}", $GoalAlignedWeight).Replace(".", "p")
        $goalTag = "goal_w$weightTag"
    }
    $RunName = "obs_counts_${goalTag}_$stamp"
}

$LogDir = Join-Path "checkpoints" $RunName
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$warmstartArgs = @()
if ($NoWarmstart) {
    $warmstartArgs += @("--warmstart-episodes", "0")
} else {
    $warmstartArgs += @(
        "--warmstart-episodes", "$WarmstartEpisodes",
        "--warmstart-epochs", "$WarmstartEpochs"
    )
}

$argsList = @(
    "-B", "scripts/train_rl.py",
    "--bin-dir", $BinDir,
    "--num-steps", "$NumSteps",
    "--total-updates", "$Updates",
    "--max-episode-ticks", "$MaxEpisodeTicks",
    "--observation-type", "entity",
    "--action-space-mode", "macro",
    "--headless",
    "--log-dir", $LogDir,
    "--no-freeze-encoder",
    "--learning-rate", "$LearningRate",
    "--clip-coef", "$ClipCoef",
    "--target-kl", "$TargetKl",
    "--ent-coef", "$EntCoef",
    "--update-epochs", "$UpdateEpochs",
    "--minibatch-size", "$MinibatchSize",
    "--teacher-kl-coef", "$TeacherKlCoef",
    "--teacher-kl-anneal-steps", "$TeacherKlAnnealSteps"
) + $warmstartArgs

if ($LoadBcActionHead) {
    $argsList += "--load-bc-action-head"
} else {
    $argsList += "--no-load-bc-action-head"
}

if ($NoGoalConditioning) {
    Write-Host "[train_best] goal_conditioning=off"
} else {
    $argsList += @(
        "--goal-conditioning",
        "--goal-aligned-weight", "$GoalAlignedWeight"
    )
}

if ($AddOpponent) {
    $argsList += "--add-opponent"
}

Write-Host "[train_best] log_dir=$LogDir"
Write-Host "[train_best] command: python $($argsList -join ' ')"

python @argsList

Write-Host "[train_best] done"
Write-Host "[train_best] csv: $(Join-Path $LogDir 'training.csv')"
