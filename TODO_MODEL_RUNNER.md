# Docker Model Runner Integration TODO Tracker

Status: In Progress 🛠️

## Steps (Sequential - Update after each completion):

- [x] 1. Create `microservices/model-runner/Dockerfile` (Ollama + CUDA optimized)
- [x] 2. Create `microservices/model-runner/entrypoint.sh` (auto-pull/deploy Llama3.1)
- [x] 3. Edit `docker-compose.yml`: Add model-runner service (port 11434, GPU, ai profile)
- [x] 4. Edit `Makefile`: Add `model-up`, `model-pull`, `model-test` targets
- [x] 5. Create this `TODO_MODEL_RUNNER.md` (done)
- [x] 6. Edit `docker/README.md`: Add Model Runner quickstart section
- [x] 7. Read & edit `mcp.json` and `microservices/mcp-server/src/mcp_tools.py`: Add LLM tools (analyze_balance, fraud_risk)
- [x] 8. Create `tests/test_model_runner.py`: Integration tests (Ollama API, MCP)
**✅ COMPLETE! All steps done.**

**Usage:**
```
make ai-up
make model-up
make model-pull
make model-test
make mcp-up
pytest tests/test_model_runner.py
```
```
make model-up
docker model pull llama3.1:latest
curl http://localhost:11434/api/tags
make mcp-up  # New tools available
```

Current progress will be updated after each step.
