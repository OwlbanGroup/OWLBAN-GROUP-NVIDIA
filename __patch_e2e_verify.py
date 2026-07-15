from pathlib import Path

p = Path("e2e_verify.py")
s = p.read_text(encoding="utf-8")

s = s.replace('add_result("Auth", success and user, msg)',
              'add_result("Auth", bool(success and user is not None), msg)')

s = s.replace('result = detector.detect_anomalies(data, threshold=150)',
              'result = detector.detect(data)')

p.write_text(s, encoding="utf-8")
print("patched e2e_verify.py")
