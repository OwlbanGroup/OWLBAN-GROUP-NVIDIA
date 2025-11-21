# Azure Deployment Fix - Thorough Testing Checklist

## Test Execution Plan

### Phase 1: Pre-Test Verification ✓
- [x] Verify Azure CLI is installed and logged in
- [x] Confirm current resource status
- [x] Document baseline state

### Phase 2: Critical Path Testing
- [ ] Test 1: Run fix_remaining_deployment.ps1
- [ ] Test 2: Verify Redis Cache creation
- [ ] Test 3: Verify Key Vault creation
- [ ] Test 4: Verify secrets storage
- [ ] Test 5: Run check_deployment_status.ps1
- [ ] Test 6: Test RUN_DEPLOYMENT_FIX.bat
- [ ] Test 7: Test RUN_CHECK_DEPLOYMENT.bat

### Phase 3: Error Handling Testing
- [ ] Test 8: Re-run script with existing resources (idempotency)
- [ ] Test 9: Verify graceful handling of existing PostgreSQL
- [ ] Test 10: Verify script continues after non-fatal errors

### Phase 4: Edge Case Testing
- [ ] Test 11: Test with different location parameter
- [ ] Test 12: Verify error messages are clear
- [ ] Test 13: Test batch file error handling

### Phase 5: Integration Testing
- [ ] Test 14: Verify all resources are accessible
- [ ] Test 15: Test Key Vault secret retrieval
- [ ] Test 16: Verify Redis connectivity
- [ ] Test 17: Verify PostgreSQL connectivity

---

## Detailed Test Cases

### Test 1: Run fix_remaining_deployment.ps1

**Command:**
```powershell
cd C:\Users\bizle\Desktop\jpmorgan_financial_apis
.\scripts\fix_remaining_deployment.ps1
```

**Expected Output:**
- Script starts with header
- Checks existing resources
- Shows PostgreSQL already exists
- Begins Redis creation (10-15 min wait)
- Begins Key Vault creation
- Stores secrets
- Shows final status

**Success Criteria:**
- No fatal errors
- Script completes without exit code 1
- Redis and Key Vault creation initiated

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 2: Verify Redis Cache Creation

**Command:**
```powershell
az redis show --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-redis --query "{name:name, state:provisioningState, location:location}" -o table
```

**Expected Output:**
```
Name                      State       Location
------------------------  ----------  ----------
jpmorgan-financial-redis  Succeeded   eastus2
```

**Success Criteria:**
- Resource exists
- State is "Succeeded" or "Creating"
- Location is correct

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 3: Verify Key Vault Creation

**Command:**
```powershell
az keyvault show --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-kv --query "{name:name, state:properties.provisioningState, location:location}" -o table
```

**Expected Output:**
```
Name                   State       Location
---------------------  ----------  ----------
jpmorgan-financial-kv  Succeeded   eastus2
```

**Success Criteria:**
- Resource exists
- State is "Succeeded"
- Location is correct

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 4: Verify Secrets Storage

**Command:**
```powershell
az keyvault secret list --vault-name jpmorgan-financial-kv --query "[].{name:name}" -o table
```

**Expected Output:**
```
Name
------------------
DatabasePassword
JWTSecret
APIKey
```

**Success Criteria:**
- All three secrets exist
- No errors accessing vault

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 5: Run check_deployment_status.ps1

**Command:**
```powershell
.\scripts\check_deployment_status.ps1
```

**Expected Output:**
```
[SUCCESS] AKS Cluster: Running
[SUCCESS] PostgreSQL: Ready
[SUCCESS] Redis Cache: Running
[SUCCESS] Key Vault: Active
[SUCCESS] ACR: Running
```

**Success Criteria:**
- All resources show SUCCESS
- No errors or warnings

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 6: Test RUN_DEPLOYMENT_FIX.bat

**Steps:**
1. Double-click RUN_DEPLOYMENT_FIX.bat
2. Observe output
3. Wait for completion

**Expected Behavior:**
- Batch file executes PowerShell script
- Shows progress messages
- Pauses at end for review

**Success Criteria:**
- Batch file runs without errors
- PowerShell script executes correctly
- User can review output before closing

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 7: Test RUN_CHECK_DEPLOYMENT.bat

**Steps:**
1. Double-click RUN_CHECK_DEPLOYMENT.bat
2. Observe output
3. Review status messages

**Expected Behavior:**
- Shows all resource statuses
- Clear SUCCESS/ERROR indicators
- Helpful next-step guidance

**Success Criteria:**
- All resources show correct status
- Output is readable and clear
- Batch file completes successfully

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 8: Idempotency Test

**Command:**
```powershell
.\scripts\fix_remaining_deployment.ps1
```

**Expected Output:**
- Script detects all resources exist
- Shows "already exists, skipping..." messages
- Completes quickly (no creation attempts)
- No errors

**Success Criteria:**
- Script runs successfully
- No duplicate resources created
- Graceful handling of existing resources

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 9: PostgreSQL Handling Test

**Verification Command:**
```powershell
az postgres flexible-server show --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-db --query "{name:name, state:state}" -o table
```

**Expected Behavior:**
- Script detects PostgreSQL exists
- Shows warning message
- Continues to next resource
- Does not exit

**Success Criteria:**
- PostgreSQL remains unchanged
- Script continues execution
- No errors thrown

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 10: Error Continuation Test

**Purpose:** Verify script continues after non-fatal errors

**Method:** Review script output for any warnings/errors and confirm script completed

**Success Criteria:**
- Script reaches end even if warnings occur
- All resources attempted
- Final summary displayed

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 11: Location Parameter Test

**Command:**
```powershell
.\scripts\fix_remaining_deployment.ps1 -Location "eastus"
```

**Expected Behavior:**
- Script accepts location parameter
- Uses specified location for new resources
- Existing resources remain unchanged

**Success Criteria:**
- Parameter is respected
- No errors from location change

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 12: Error Message Clarity Test

**Method:** Review all output messages during testing

**Evaluation Criteria:**
- Error messages are clear and actionable
- Success messages are encouraging
- Warnings are informative
- Color coding is appropriate

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 13: Batch File Error Handling

**Method:** Test batch files with various scenarios

**Test Cases:**
- Run from different directory
- Run with PowerShell execution policy restrictions
- Run when Azure CLI not logged in

**Success Criteria:**
- Appropriate error messages
- Graceful failure
- User guidance provided

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 14: Resource Accessibility

**Commands:**
```powershell
# Test AKS
az aks get-credentials --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-aks --overwrite-existing
kubectl get nodes

# Test ACR
az acr login --name jpmorganfinancialacr

# Test PostgreSQL
az postgres flexible-server show --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-db
```

**Success Criteria:**
- All resources are accessible
- Credentials work correctly
- No permission errors

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 15: Key Vault Secret Retrieval

**Command:**
```powershell
az keyvault secret show --vault-name jpmorgan-financial-kv --name "DatabasePassword" --query "value" -o tsv
az keyvault secret show --vault-name jpmorgan-financial-kv --name "JWTSecret" --query "value" -o tsv
az keyvault secret show --vault-name jpmorgan-financial-kv --name "APIKey" --query "value" -o tsv
```

**Success Criteria:**
- All secrets can be retrieved
- Values are non-empty
- No access errors

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 16: Redis Connectivity

**Command:**
```powershell
# Get Redis connection info
az redis show --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-redis --query "{hostname:hostName, sslPort:sslPort, port:port}" -o table

# Get access keys
az redis list-keys --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-redis --query "{primaryKey:primaryKey}" -o table
```

**Success Criteria:**
- Connection details retrieved
- Access keys available
- Redis is accessible

**Actual Result:**
```
[To be filled during testing]
```

---

### Test 17: PostgreSQL Connectivity

**Command:**
```powershell
# Get connection string
az postgres flexible-server show --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-db --query "{fqdn:fullyQualifiedDomainName, state:state}" -o table

# List databases
az postgres flexible-server db list --resource-group jpmorgan-financial-apis-rg --server-name jpmorgan-financial-db --query "[].{name:name}" -o table
```

**Success Criteria:**
- Server is accessible
- Database exists
- Connection info is correct

**Actual Result:**
```
[To be filled during testing]
```

---

## Test Summary

### Tests Passed: 0/17
### Tests Failed: 0/17
### Tests Skipped: 0/17

### Critical Issues Found:
```
[To be documented during testing]
```

### Non-Critical Issues Found:
```
[To be documented during testing]
```

### Recommendations:
```
[To be documented after testing]
```

---

## Sign-Off

**Tester:** ___________________
**Date:** ___________________
**Overall Status:** [ ] PASS [ ] FAIL [ ] NEEDS REVISION

**Notes:**
```
[Additional notes and observations]
