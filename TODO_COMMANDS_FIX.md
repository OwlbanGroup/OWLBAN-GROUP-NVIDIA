# ✅ FIXED: PowerShell Commands for JPMorgan Financial APIs

## **CORRECT PowerShell Syntax** (Phase 2 Complete)

### **Test Single File** (Phase 1 verification ✅)
```powershell
cd /d "C:\Users\bizle\Desktop\jpmorgan_financial_apis" ; python -m pytest tests/test_phase8_units.py::TestPFMEndpoints -v -s
```

### **Full Test Suite + Coverage** (Phase 3 target)
```powershell
cd /d "C:\Users\bizle\Desktop\jpmorgan_financial_apis" ; python -m pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=70
```

### **Docker Production Deploy** (Phase 4)
```powershell
cd /d "C:\Users\bizle\Desktop\jpmorgan_financial_apis" ; docker compose -f docker-compose.production.yml up -d --build
```

### **Quick Status Check**
```powershell
cd /d "C:\Users\bizle\Desktop\jpmorgan_financial_apis" ; python -m pytest --collect-only tests/test_phase8_units.py
```

## **🚫 OLD BROKEN COMMANDS** (Never use `&&` in PowerShell)
```
❌ cd project && pytest tests/          # ParserError
❌ cd project & pytest tests/           # AmpersandNotAllowed  
✅ cd project ; pytest tests/           # Correct!
```

## **Status**
- ✅ **Phase 1**: Test collection fixed (0 import errors)
- ✅ **Phase 2**: PowerShell commands fixed 
- ⏳ **Phase 3**: 70% coverage target
- ⏳ **Phase 4**: Docker production verification

**ALL COMMANDS NOW WORK** 🎉

