param(
    [string]$Message = "Update TextAnalysis platform"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  GitHub (mshungo/TextAnalysis) 自動プッシュ" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`n1. ファイルのステージング中..." -ForegroundColor Yellow
git add .

$status = git status --porcelain
if (-not $status) {
    Write-Host "変更されたファイルはありません。最新の状態です。" -ForegroundColor Green
    exit 0
}

Write-Host "`n2. コミットを作成中: '$Message'..." -ForegroundColor Yellow
git commit -m $Message

Write-Host "`n3. GitHubへプッシュ中 (origin main)..." -ForegroundColor Yellow
git push origin main

Write-Host "`n>>> GitHubへの公開・更新が正常に完了しました！" -ForegroundColor Green
