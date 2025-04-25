class OwlbanAI:
    def __init__(self):
        self.models_loaded = False

    def load_models(self):
        print("Loading OWLBAN AI models...")
        # Simulate model loading
        self.models_loaded = True
        print("Models loaded successfully.")

    def run_inference(self, data):
        if not self.models_loaded:
            raise Exception("Models not loaded.")
        print(f"Running inference on data: {data}")
        # Simulate inference result
        return {"prediction": "positive", "confidence": 0.95}
