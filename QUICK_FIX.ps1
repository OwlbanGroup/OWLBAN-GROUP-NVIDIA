# JPMorgan Financial APIs - One-Command Quick Fix
# This is the simplest way to fix all Docker issues

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   JPMorgan Financial APIs - Docker Quick Fix              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  ✓ Backup all your data safely" -ForegroundColor Green
Write-Host "  ✓ Fix PostgreSQL version mismatch" -ForegroundColor Green
Write-Host "  ✓ Fix AlertManager configuration" -ForegroundColor Green
Write-Host "  ✓ Restart all services properly" -ForegroundColor Green
Write-Host ""

Write-Host "Estimated time: 5-10 minutes" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "Ready to proceed? (yes/no)"

if ($response -eq "yes" -or $response -eq "y") {
    Write-Host ""
    Write-Host "Starting automated fix process..." -ForegroundColor Green
    Write-Host ""
    
    # Run the backup and fix script
    & ".\backup_and_fix_docker.ps1"
    
} else {
    Write-Host ""
    Write-Host "Fix cancelled." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "When you're ready, run:" -ForegroundColor Cyan
    Write-Host "  .\QUICK_FIX.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or for more control:" -ForegroundColor Cyan
    Write-Host "  .\backup_and_fix_docker.ps1" -ForegroundColor White
    Write-Host ""
}
