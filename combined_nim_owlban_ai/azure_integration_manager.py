"""Compatibility shim for Azure integration imports."""


class AzureQuantumIntegrationManager:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def create_compute_cluster(self, *args, **kwargs):
        return {"status": "mock", "cluster": args[0] if args else "default"}

    def submit_training_job(self, *args, **kwargs):
        return {"status": "mock"}

    def deploy_model(self, *args, **kwargs):
        return {"status": "mock"}
