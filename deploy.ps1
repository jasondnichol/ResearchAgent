# deploy.ps1 - Deploy ResearchAgent files to EC2 and restart bot

$Key = "C:\Users\TradingBot\tradingbot-key.pem"
$Host_ = "ubuntu@184.72.84.30"
$RemoteDir = "/home/ubuntu/ResearchAgent/"

$Files = @(
    "integrated_switcher.py",
    "notify.py",
    "daily_summary.py",
    "market_regime.py",
    "williams_r_strategy.py",
    "research_agent_v2.py",
    "strategy_library.py",
    "strategy_library.json",
    "CLAUDE.md"
)

Write-Host "=== Deploying to EC2 ===" -ForegroundColor Cyan

foreach ($File in $Files) {
    $Path = Join-Path $PSScriptRoot $File
    if (Test-Path $Path) {
        Write-Host "Uploading $File..." -ForegroundColor Yellow
        scp -i $Key $Path "${Host_}:${RemoteDir}"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED to upload $File" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "WARNING: $File not found, skipping" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Restarting bot ===" -ForegroundColor Cyan

ssh -i $Key $Host_ "cd /home/ubuntu/ResearchAgent && screen -S tradingbot -X stuff `$'\003' && sleep 2 && screen -S tradingbot -X stuff 'python integrated_switcher.py\n'"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Bot restarted successfully!" -ForegroundColor Green
} else {
    Write-Host "Failed to restart bot" -ForegroundColor Red
    exit 1
}

Write-Host "=== Deploy complete ===" -ForegroundColor Green
