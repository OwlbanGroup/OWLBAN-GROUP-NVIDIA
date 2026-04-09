.PHONY: help up down logs build test clean ai-up ai-down gpu-test docs buildx-bootstrap build-offload buildx-push

# Colors
GREEN = \033[0;32m
NC = \033[0m # No Color

PROJECT_NAME = jpmorgan-financial-apis
DC = docker compose
DCD = docker compose -f docker-compose.dev.yml  # microservices dev

help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: .env ## Start dev stack (API Gateway + infra)
	@echo "$(GREEN)🚀 Starting $(PROJECT_NAME) dev stack...$(NC)"
	$(DC) up -d
	@echo "$(GREEN)✅ Gateway ready: http://localhost:8000/health$(NC)"
	@echo "$(GREEN)📊 Swagger: http://localhost:8000/docs$(NC)"
	@$(DC) ps

ai-up: .env ## Start AI/ML stack with GPU
	@echo "$(GREEN)🤖 Starting AI/ML stack with GPU...$(NC)"
	$(DC) --profile ai up -d
	@echo "$(GREEN)✅ ML Service GPU: http://localhost:8002/health$(NC)"
	@$(DC) ps ml-service traction-service

down: ## Stop stack
	$(DC) down -v

logs: ## Tail logs
	$(DC) logs -f gateway

build: ## Build images
	$(DC) build --no-cache

test: ## Run pytest in container
	$(DC) exec gateway pytest tests/ -v --cov=jpmorgan_financial_apis/ --cov-report=html

gpu-test: ## Test GPU availability (nvidia-smi)
	$(DC) exec ml-service nvidia-smi || echo "No GPU detected - install NVIDIA Container Toolkit"

lint: ## Lint code
	$(DC) exec gateway flake8 . || true
	$(DC) exec gateway black --check .

clean: down ## Clean volumes/images
	docker system prune -af --volumes

docs: ## Generate coverage report
	$(DC) exec gateway pytest --cov-report=html
	@echo "📄 Coverage report: $$(pwd)/htmlcov/index.html"

mcp-up: .env ## Start MCP server stack
	@echo "$(GREEN)🔌 Starting MCP Server...$(NC)"
	$(DC) up mcp-server -d --build
	@echo "$(GREEN)✅ MCP Server: http://localhost:8080/health$(NC)"
	@echo "$(GREEN)📋 Tools: http://localhost:8080/mcp/tools$(NC)"
	@$(DC) ps mcp-server

mcp-logs: ## MCP logs
	$(DC) logs -f mcp-server

mcp-test: ## Test MCP
	$(DC) exec mcp-server curl http://localhost:8080/mcp/tools || echo "MCP tools list"

mcp-down: ## Stop MCP
	$(DC) stop mcp-server

model-up: .env ## Start Model Runner LLM (GPU)
	@echo "$(GREEN)🤖 Starting Model Runner (Ollama GPU)...$(NC)"
	$(DC) --profile ai up model-runner -d --build
	@echo "$(GREEN)✅ Model Runner: http://localhost:11434/api/tags$(NC)"
	@echo "$(GREEN)📥 Pull model: make model-pull$(NC)"
	@$(DC) ps model-runner

model-logs: ## Model Runner logs
	$(DC) logs -f model-runner

model-test: ## Test Model Runner (health + nvidia-smi)
	$(DC) exec model-runner curl -f http://localhost:11434/api/tags || true
	$(DC) exec model-runner nvidia-smi || echo "GPU check"
	@echo "$(GREEN)💬 Test chat: curl -X POST http://localhost:11434/api/chat \\$$' -d '{\"model\":\"llama3.1\",\"messages\":[{\"role\":\"user\",\"content\":\"Analyze JPMorgan stock risk.\"}] }'$(NC)"

model-down: ## Stop Model Runner
	$(DC) stop model-runner

model-pull: ## Pull LLM model (make model-pull MODEL=phi3:latest)
	docker model pull ${MODEL:-llama3.1:latest}

scout: ## Docker Scout analysis (requires login)
	@if ! docker scout version &> /dev/null; then \
		echo "$(GREEN)Installing Docker Scout...$(NC)"; \
		curl -sSfL https://cli.docker.com/docker-scout.sh | sh -s -- install-cli; \
	fi
	docker scout quickstart . --analyze
	@echo "$(GREEN)✅ Docker Scout complete!$(NC)"

k8s-up: ## Deploy to Kubernetes (requires kubectl, kind/minikube)
	kubectl apply -f deployment/k8s-deployment.yaml
	kubectl get pods -n jpmorgan

k8s-down: ## Cleanup Kubernetes
	kubectl delete -f deployment/k8s-deployment.yaml --ignore-not-found

# Docker Offload / Buildx targets
buildx-bootstrap: ## Setup Buildx builder for offload
	@echo "$(GREEN)🔧 Bootstrapping Buildx offload builder...$(NC)"
	docker buildx create --name offload-builder --driver docker-container --bootstrap --config docker/buildkitd.toml || docker buildx use offload-builder
	docker buildx inspect --bootstrap
	@echo "$(GREEN)✅ Offload builder ready$(NC)"

build-offload: buildx-bootstrap ## Build all images with offload (parallel GPU/CUDA)
	@echo "$(GREEN)🚀 Building OFFLOAD group (ml-cuda, model-runner, mcp, traction)...$(NC)"
	docker buildx bake docker-compose.build.hcl --file docker/docker-compose.build.hcl#offload --progress=plain --load
	@echo "$(GREEN)✅ Offload build complete!$(NC)"

buildx-push: build-offload ## Push offload images to registry
	docker buildx bake docker-compose.build.hcl --file docker/docker-compose.build.hcl#offload --push

registry-login: ## Login to Docker Hub (docker login)
	@echo "$(GREEN)🔐 Docker Hub login...$(NC)"
	@docker login -u ${DOCKER_USER:-$$DOCKER_USER} -p ${DOCKER_PASS:-$$DOCKER_PASS}
	@echo "$(GREEN)✅ Logged in to Docker Hub!$(NC)"

image-push: registry-login build-offload ## Build & push all images (Hub/Registry API)
	@echo "$(GREEN)📤 Pushing images: gateway ml mcp model-runner...$(NC)"
	docker tag jpmorgan_gateway:latest ${DOCKER_USER:-$$DOCKER_USER}/jpmorgan-financial-gateway:latest
	docker tag jpmorgan_mcp_server:latest ${DOCKER_USER:-$$DOCKER_USER}/jpmorgan-mcp-server:latest
	docker tag jpmorgan_model_runner:latest ${DOCKER_USER:-$$DOCKER_USER}/jpmorgan-model-runner:latest
	docker push ${DOCKER_USER:-$$DOCKER_USER}/jpmorgan-financial-gateway:latest
	docker push ${DOCKER_USER:-$$DOCKER_USER}/jpmorgan-mcp-server:latest
	docker push ${DOCKER_USER:-$$DOCKER_USER}/jpmorgan-model-runner:latest
	@echo "$(GREEN)✅ Images pushed! Refs: https://hub.docker.com/r/$${DOCKER_USER}/jpmorgan-financial-gateway$(NC)"

mcp-catalog: ## Add to Docker MCP Catalog
	docker mcp catalog add mcp.json
	@echo "$(GREEN)✅ Added to MCP Catalog!$(NC)"

docker-full-test: up ## Full Docker reference test suite (health + endpoints)
	@echo "$(GREEN)🧪 Running Docker full test suite...$(NC)"
	@sleep 10  # Wait startup
	@curl -f http://localhost:8000/health || (echo "❌ Gateway failed"; exit 1)
	@curl -f http://localhost:8080/health || (echo "⚠️ MCP optional"; true)
	@curl -f http://localhost:11434/api/tags || (echo "⚠️ Model Runner optional"; true)
	@echo "$(GREEN)✅ Gateway healthy!$(NC)"
	$(DC) ps
	python3 docker/engine-api-example.py || echo "Engine API example (run 'pip install docker')"

desktop: .env ## Docker Desktop GUI launch
	@echo "$(GREEN)🖥️  Launching Docker Desktop GUI stack...$(NC)"
	$(DC) -f docker-compose.desktop.yml up -d
	@echo "$(GREEN)✅ Stack in Docker Desktop > Compose > jpm-finance-gui$(NC)"
	@echo "$(GREEN)🖥️  Double-click start-desktop.bat for full GUI + browsers$(NC)"
	$(DC) -f docker-compose.desktop.yml ps

.env:
	cp .env.example .env
	@echo "$(GREEN)📝 .env created from .env.example - EDIT SECRETS!$(NC)"

