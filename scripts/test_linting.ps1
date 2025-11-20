$results = Invoke-ScriptAnalyzer -Path "fix_remaining_deployment.ps1"
$unusedVarWarning = $results | Where-Object { $_.RuleName -eq 'PSUseDeclaredVarsMoreThanAssignments' }

if ($unusedVarWarning) {
    Write-Host "FAILED: PSUseDeclaredVarsMoreThanAssignments warning still exists" -ForegroundColor Red
    $unusedVarWarning | Format-List
} else {
    Write-Host "SUCCESS: No PSUseDeclaredVarsMoreThanAssignments warnings found!" -ForegroundColor Green
}

Write-Host "`nAll PSScriptAnalyzer warnings:" -ForegroundColor Cyan
$results | Select-Object RuleName, Line, Message | Format-Table -AutoSize
