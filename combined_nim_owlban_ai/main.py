from combined_nim_owlban_ai.nim import NimManager
from combined_nim_owlban_ai.owlban_ai import OwlbanAI
from combined_nim_owlban_ai.integration import CombinedSystem

def main():
    # Initialize NVIDIA NIM manager
    nim_manager = NimManager()
    nim_manager.initialize()

    # Initialize OWLBAN AI system
    owlban_ai = OwlbanAI()
    owlban_ai.load_models()

    # Initialize combined system
    combined_system = CombinedSystem()
    combined_system.initialize()
    combined_system.start_operations()

if __name__ == "__main__":
    main()
