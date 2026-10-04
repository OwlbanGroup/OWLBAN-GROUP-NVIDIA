import sys, traceback

target = sys.argv[1] if len(sys.argv) > 1 else "api_server"
out_path = "probe_out.txt"
mode = "a" if len(sys.argv) > 2 else "w"
buf = []
try:
    __import__(target)
    buf.append("IMPORT_OK: " + target)
except Exception:
    buf.append("IMPORT_FAIL: " + target)
    buf.append(traceback.format_exc())
with open(out_path, mode, encoding="utf-8") as f:
    f.write("\n".join(buf) + "\n")

