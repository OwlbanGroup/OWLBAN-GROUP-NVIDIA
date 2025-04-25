from nim import NimManager
from owlban_ai import OwlbanAI
from integration import IntegrationManager

def main():
    # Initialize NVIDIA NIM manager
    nim_manager = NimManager()
    nim_manager.initialize()

    # Initialize OWLBAN AI system
    owlban_ai = OwlbanAI()
    owlban_ai.load_models()

    # Initialize integration manager
    integration_manager = IntegrationManager(nim_manager, owlban_ai)
    integration_manager.run()

if __name__ == "__main__":
    main()
