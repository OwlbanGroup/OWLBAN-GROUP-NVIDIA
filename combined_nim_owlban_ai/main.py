"""
Main entry point for the Combined NIM OWLBAN AI system.

This module initializes and starts the integrated NVIDIA NIM and OWLBAN AI components,
providing a unified quantum financial and AI operations platform.
"""

from combined_nim_owlban_ai.nim import NimManager
from combined_nim_owlban_ai.owlban_ai import OwlbanAI
from combined_nim_owlban_ai import CombinedSystem


def main():
    """
    Initialize and start the combined NIM OWLBAN AI system.

    This function sets up the NVIDIA NIM manager, OWLBAN AI system,
    and the integrated combined system for quantum financial operations.
    """
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
