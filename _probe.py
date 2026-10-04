import sys, traceback, logging, warnings

# Silence the noisy "Qiskit not available ..." logging.warning emitted at
# import time by combined_nim_owlban_ai/integration.py. We only want the
# real import result written to the file.
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

targets = sys.argv[1:] if len(sys.argv) > 1 else ["api_server"]
buf = []
for target in targets:
    try:
        __import__(target)
        buf.append("IMPORT_OK: " + target)
    except Exception:
        buf.append("IMPORT_FAIL: " + target)
        buf.append(traceback.format_exc())
with open("probe_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(buf) + "\n")



