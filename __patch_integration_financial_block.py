from pathlib import Path

p = Path("combined_nim_owlban_ai/integration.py")
lines = p.read_text(encoding="utf-8").splitlines(True)

# Find the exact region starting at the sentinel comment
start = None
for i, l in enumerate(lines):
    if l.strip() == "# Financial Integration":
        start = i
        break

if start is None:
    raise SystemExit("Could not find '# Financial Integration' sentinel")

# End right before the next import we saw in the failing region
end = None
for j in range(start + 1, len(lines)):
    if lines[j].startswith("from new_products.infrastructure_optimizer import InfrastructureOptimizer"):
        end = j
        break

if end is None:
    raise SystemExit("Could not find end sentinel import")

replacement = [
    "# Financial Integration\n",
    "try:\n",
    "    import bloombergl  # type: ignore\n",
    "except ImportError:\n",
    "    bloombergl = None  # type: ignore\n",
    "\n",
    "logging.warning(\"Numba CUDA disabled due to compatibility issues\")\n",
    "\n",
]

lines[start:end] = replacement
p.write_text("".join(lines), encoding="utf-8")
print("Patched integration.py financial integration block")
