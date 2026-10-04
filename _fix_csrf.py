import re
p = "middleware/csrf.py"
s = open(p, encoding="utf-8").read()
# Normalize the leading whitespace of the class-level comment block that got
# mis-indented during editing back to 4 spaces (class-body level).
s = re.sub(r"(?m)^[ \t]*# Bearer/Basic-token-authenticated API routes do not need cookie-CSRF$",
           "    # Bearer/Basic-token-authenticated API routes do not need cookie-CSRF", s)
s = re.sub(r"(?m)^[ \t]*# protection \(the browser does not auto-attach bearer tokens",
           "    # protection (the browser does not auto-attach bearer tokens", s)
s = re.sub(r"(?m)^[ \t]*# is likewise not a cookie-based flow\. Only cookie",
           "    # is likewise not a cookie-based flow. Only cookie", s)
s = re.sub(r"(?m)^[ \t]*# web surfaces \(e\.g\. OWLBAN GROUP site",
           "    # web surfaces (e.g. OWLBAN GROUP site", s)
s = re.sub(r"(?m)^[ \t]*# the double-submit token\.$",
           "    # the double-submit token.", s)
open(p, "w", encoding="utf-8").write(s)
print("csrf.py comment block normalized")
