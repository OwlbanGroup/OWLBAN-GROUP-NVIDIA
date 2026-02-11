# TODO: Fix FastAPI App Import and Runtime Errors

## Step 1: Fix contextmanager import in database_fixed.py
- Change @contextmanager to @asynccontextmanager in AsyncDatabaseManager.get_session method

## Step 2: Fix Limiter initialization in app_async.py
- Change limiter = Limiter(app, key_func=get_remote_address) to limiter = Limiter(key_func=get_remote_address)

## Step 3: Check and fix telemetry_handler_new import
- Ensure src/telemetry_handler_new.py exists or create it if missing
- Verify imports in app_async.py

## Step 4: Test app import
- Run python -c "import app_async; print('FastAPI app imports successfully')"

## Step 5: Test app startup
- Run uvicorn app_async:app --host 0.0.0.0 --port 8000 and check for errors

## Step 6: Test endpoints
- Test /health, /, user/register, user/login endpoints
