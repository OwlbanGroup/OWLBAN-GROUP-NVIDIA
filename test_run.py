import time
from combined_nim_owlban_ai.integration import CombinedSystem

def run_test(iterations=7000):
    system = CombinedSystem()
    system.initialize()
    start_time = time.time()
    for i in range(iterations):
        print(f"Running test iteration {i+1}/{iterations}")
        system.start_operations()
    end_time = time.time()
    print(f"Completed {iterations} iterations in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_test()
