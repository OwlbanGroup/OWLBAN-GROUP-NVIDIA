from pathlib import Path

p = Path("combined_nim_owlban_ai/rapids_integration.py")
s = p.read_text(encoding="utf-8")

old = "import pandas as pd"
if old not in s:
    raise SystemExit("Target 'import pandas as pd' not found")

new = """try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:
    pd = None  # type: ignore"""

s2 = s.replace(old, new, 1)
p.write_text(s2, encoding="utf-8")
print("Patched rapids_integration.py: pandas is now optional")
