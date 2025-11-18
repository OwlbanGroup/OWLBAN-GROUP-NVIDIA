# Dashboard Linting Fixes - Progress Tracker

## Completed ✅
- [x] Created `shared/config.py` with Settings class
- [x] Created `shared/monitoring.py` with MetricsCollector
- [x] Updated `shared/__init__.py` to export new modules

## In Progress 🔄
- [ ] Fix main.py (200+ issues)

## Main.py Fixes Required

### Import Fixes (Priority 1)
- [ ] Change relative imports (`..shared`) to absolute imports (`shared`)
- [ ] Remove unused imports (List, Response, Depends, etc.)
- [ ] Fix import order (standard → third-party → local)
- [ ] Remove non-existent TrustProxyMiddleware import
- [ ] Add type stub installation note for jose, plotly

### Type Annotation Fixes (Priority 2)
- [ ] Fix SQLAlchemy Base type issues in models
- [ ] Add proper type hints for all function parameters
- [ ] Fix Optional parameter defaults
- [ ] Add return type annotations
- [ ] Fix Collection[str] type issues

### Code Quality Fixes (Priority 3)
- [ ] Remove 150+ trailing whitespace occurrences
- [ ] Fix 30+ lines exceeding 100 characters
- [ ] Add docstrings for DashboardConfig class
- [ ] Add docstrings for Widget class
- [ ] Add docstrings for ConnectionManager methods
- [ ] Fix bare except clauses
- [ ] Add proper exception handling with `raise from`

### Function-Specific Fixes (Priority 4)
- [ ] Fix unused 'request' parameters (prefix with _ or remove)
- [ ] Fix broad Exception catching
- [ ] Add missing docstrings for helper functions

## Testing After Fixes
- [ ] Run mypy type checking
- [ ] Run pylint code quality check
- [ ] Test imports work correctly
- [ ] Verify application still runs
- [ ] Install type stubs: `pip install types-python-jose`

## Notes
- File currently has 1047 lines (47 over limit)
- Consider splitting into multiple modules if needed
- All fixes maintain backward compatibility
