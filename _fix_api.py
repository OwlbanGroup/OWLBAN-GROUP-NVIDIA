"""One-off mechanical fixer for api_server.py lint issues."""
import re

path = 'api_server.py'
src = open(path, encoding='utf-8').read()

# 1. raise-missing-from: convert
#   except Exception:\n ... logger.exception("X")\n raise HTTPException(...)
# to `except Exception as exc:` ... `raise ... from exc`
src = src.replace(
    'except Exception:\n        logger.exception(',
    'except Exception as exc:  # pylint: disable=broad-exception-caught\n'
    '        logger.exception(')
src = re.sub(
    r'(raise HTTPException\(status_code=500, detail="[^"]+")\)\n',
    r'\1) from exc\n',
    src)

# 2. % formatting -> f-string
src = src.replace(
    '"message": "Revenue optimization started with %d iterations" % request.iterations,',
    '"message": f"Revenue optimization started with {request.iterations} iterations",')

# 3. Reflow long @fastapi_app route decorators (move responses=... to own line)
def reflow_decorator(m):
    method, route, responses = m.group(1), m.group(2), m.group(3)
    return ('@fastapi_app.%s(\n    %s,\n    responses=%s)' %
            (method, route, responses))

src = re.sub(
    r'@fastapi_app\.(get|post|delete|put)\((".*?"), (responses=\{[^}]*\{[^}]*\}[^}]*\})\)',
    reflow_decorator, src)

# 4. Missing class docstrings for pydantic models
class_docs = {
    'class RevenueOptimizationRequest(BaseModel):':
        '    """Request payload for quantum revenue optimization."""',
    'class InferenceRequest(BaseModel):':
        '    """Request payload for combined-system inference."""',
    'class SystemStatus(BaseModel):':
        '    """Aggregated health status of all API services."""',
    'class LogEntry(BaseModel):':
        '    """A single application log entry."""',
    'class RegisterRequest(BaseModel):':
        '    """User registration payload."""',
    'class LoginRequest(BaseModel):':
        '    """User login payload with optional MFA code."""',
    'class TokenResponse(BaseModel):':
        '    """JWT access/refresh token pair."""',
    'class ResetRequest(BaseModel):':
        '    """Password reset request payload."""',
    'class ResetPasswordRequest(BaseModel):':
        '    """Password reset confirmation payload."""',
    'class APIKeyRequest(BaseModel):':
        '    """API key creation payload."""',
    'class MFACodeRequest(BaseModel):':
        '    """TOTP MFA code payload."""',
    'class UserProfile(BaseModel):':
        '    """Public user profile fields."""',
}
lines = src.split('\n')
out = []
for line in lines:
    out.append(line)
    stripped = line.rstrip()
    if stripped in class_docs and (not out or '"""' not in out[-2]):
        out.append(class_docs[stripped])
src = '\n'.join(out)

# 5. Missing function docstrings (route handlers)
func_docs = {
    'async def root():': '    """Root service banner."""',
    'async def health_check():': '    """Liveness probe endpoint."""',
    'async def get_system_status():': '    """Full system status across all services."""',
    'async def get_catalog_summary():': '    """Summarize the NGC model catalog."""',
    'async def search_catalog(query: str):': '    """Search the NGC model catalog."""',
    'async def optimize_revenue(request: RevenueOptimizationRequest,'
    ' background_tasks: BackgroundTasks):':
        '    """Start a revenue optimization run in the background."""',
    'async def get_current_profit():': '    """Return the current optimized profit."""',
    'async def run_inference(request: InferenceRequest):':
        '    """Run combined-system inference."""',
    'async def rl_learn(state: List[float], action: str, reward: float,'
    ' next_state: List[float]):':
        '    """Train the reinforcement learning agent on one transition."""',
    'async def get_rl_action(state: List[float]):':
        '    """Select the next RL action for a state."""',
    'async def get_gpu_status():': '    """Return GPU resource status."""',
    'async def get_quantum_portfolio():':
        '    """Run quantum portfolio optimization."""',
    'async def get_quantum_risk():': '    """Run quantum risk analysis."""',
    'async def predict_market(symbol: str):':
        '    """Predict market movement for a symbol."""',
    'async def prometheus_metrics():': '',  # already has docstring
    'async def mfa_setup(user=Depends(get_current_user)):': '',  # has one
}
out = []
i = 0
while i < len(lines := src.split('\n')):
    line = lines[i]
    out.append(line)
    stripped = line.strip()
    matched = None
    for sig, doc in func_docs.items():
        if stripped.startswith(sig.split('(')[0]) and stripped == sig:
            matched = doc
            break
    if matched and (i + 1 >= len(lines) or '"""' not in lines[i + 1]):
        out.append(matched)
    i += 1
src = '\n'.join(out)

# 6. E302/E305: ensure exactly two blank lines before top-level
#    def/class/decorator/assignment statements.
out = []
def is_top(stmt):
    return bool(re.match(r'^(def |class |async def |@|[A-Za-z_][A-Za-z0-9_.]* =|# )',
                         stmt))

i = 0
n = len(lines := src.split('\n'))
while i < n:
    line = lines[i]
    if (i > 0 and line and not line.startswith((' ', '\t', ')'))
            and re.match(r'^(def |class |async def |@|[A-Za-z_][A-Za-z0-9_]* =)',
                         line)
            and not (out and out[-1] == '' and len(out) >= 2 and out[-2] == '')):
        # count trailing blanks
        blanks = 0
        while blanks < len(out) and out[-1 - blanks] == '':
            blanks += 1
        if blanks == 1:
            out.append('')  # add second blank line
        elif blanks == 0 and out and out[-1].strip() != '':
            out.extend(['', ''])
    out.append(line)
    i += 1
src = '\n'.join(out)

open(path, 'w', encoding='utf-8').write(src)
print('done')
