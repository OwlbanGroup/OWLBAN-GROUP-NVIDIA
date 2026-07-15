from pathlib import Path

p = Path("combined_nim_owlban_ai/nim.py")
s = p.read_text(encoding="utf-8")

s = s.replace("import psutil\n", "try:\n    import psutil  # type: ignore\nexcept ModuleNotFoundError:\n    psutil = None  # type: ignore\n")

s = s.replace("import GPUtil\n", "try:\n    import GPUtil  # type: ignore\nexcept ModuleNotFoundError:\n    GPUtil = None  # type: ignore\n")

# Guard psutil usage in get_resource_status()
needle = "cpu_percent = psutil.cpu_percent(interval=1)\n        memory = psutil.virtual_memory()\n        status.update({"
if needle in s:
    repl = ("if psutil is None:\n"
            "            cpu_percent = 0.0\n"
            "            memory = type('M', (), {'percent': 0.0})()\n"
            "        else:\n"
            "            cpu_percent = psutil.cpu_percent(interval=1)\n"
            "            memory = psutil.virtual_memory()\n"
            "        status.update({")
    s = s.replace(needle, repl)
else:
    # If structure differs, do not apply the guard; file will still import but metrics may fail.
    pass

p.write_text(s, encoding="utf-8")
print("Patched nim.py optional psutil/GPUtil")
