function platforms = ["linux/amd64", "linux/arm64"]

target "default" {
  platforms   = platforms
  dockerfile  = "Dockerfile"
  context     = "./microservices/mcp-server"
  tags = [
    "jpmorgan-financial-apis-mcp:latest",
    "jpmorgan-financial-apis-mcp:offload"
  ]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
}

target "ml-cuda" {
  platforms  = ["linux/amd64"]
  dockerfile = "Dockerfile.cuda"
  context    = "./microservices/ml"
  args = {
    PYTHON_VERSION = "3.12"
  }
  tags = [
    "jpmorgan-financial-apis-ml-cuda:latest",
    "jpmorgan-financial-apis-ml-cuda:offload"
  ]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
}

target "model-runner" {
  platforms   = ["linux/amd64"]
  dockerfile  = "Dockerfile"
  context     = "./microservices/model-runner"
  tags = [
    "jpmorgan-financial-apis-model-runner:latest",
    "jpmorgan-financial-apis-model-runner:offload"
  ]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
}

target "traction" {
  platforms   = platforms
  dockerfile  = "Dockerfile.optimized"
  context     = "./microservices/traction"
  tags = [
    "jpmorgan-financial-apis-traction:latest",
    "jpmorgan-financial-apis-traction:offload"
  ]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
}

group "offload" {
  targets = ["default", "ml-cuda", "model-runner", "traction"]
}

