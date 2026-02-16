# Pylint Fix Plan

## Current State

- Pylint rating: 7.44/10
- Target: 10.00/10

## Issues to Fix

### 1. Import Order (C0413) - 31 instances

- Move sys.path manipulation to the very top of the file
- Reorganize imports to be at the top of the module
- Add proper pylint disable comments where needed

### 2. Unnecessary "else" after "return" (R1705) - 5 instances

- Line 357: receive_telemetry function
- Line 574: receive_telemetry_batch function
- Line 1031: create_issue function
- Line 1080: login function
- Line 1156: public_register function

### 3. Too many return statements (R0911) - 2 instances

- Line 318: receive_telemetry function (7/6)
- Line 1108: login function (7/6)

## Implementation Steps

1. [ ] Move sys.path.insert to the very top of app.py
2. [ ] Reorganize imports to be at the top
3. [ ] Fix unnecessary else/return patterns (5 instances)
4. [ ] Refactor functions with too many returns (2 instances)
5. [ ] Run pylint to verify 10.00/10 rating
