$files = @('test_api_server_status','test_banking_applications','test_e2e_smoke','test_ngc_catalog','test_quantum_ai','test_quantum_machine_learning')
$env:API_PASSWORD = 'testpass123'
Set-Location $PSScriptRoot
Remove-Item per_file_results.txt -ErrorAction SilentlyContinue
foreach ($f in $files) {
    $out = python -m pytest ("tests/" + $f + ".py") -q --tb=line --no-header --timeout=120 --timeout-method=thread 2>&1 | Out-String
    $line = ($out -split "`n" | Where-Object { $_ -match 'passed|failed|error' } | Select-Object -First 1)
    Add-Content -Path per_file_results.txt -Value ($f + " => " + $line)
}
Add-Content -Path per_file_results.txt -Value "ALL_DONE"