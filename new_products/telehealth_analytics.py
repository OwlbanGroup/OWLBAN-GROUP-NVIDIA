class TelehealthAnalytics:
    def __init__(self, nim_manager, owlban_ai):
        self.nim_manager = nim_manager
        self.owlban_ai = owlban_ai

    def analyze_patient_data(self, patient_data):
        print("Analyzing patient data with AI...")
        if not self.owlban_ai.models_loaded:
            self.owlban_ai.load_models()
        result = self.owlban_ai.run_inference(patient_data)
        print(f"Patient data analysis result: {result}")

    def monitor_infrastructure(self):
        resource_status = self.nim_manager.get_resource_status()
        print(f"Telehealth infrastructure status: {resource_status}")
