from human_ai_collaboration.environment import HumanAICollaborationEnvironment

class CollaborationManager:
    def __init__(self):
        self.environment = HumanAICollaborationEnvironment()

    def assign_tasks(self, human_tasks, ai_tasks):
        for task in human_tasks:
            self.environment.add_human_task(task)
        for task in ai_tasks:
            self.environment.add_ai_task(task)

    def allocate_resources(self, resources):
        for name, resource in resources.items():
            self.environment.allocate_resource(name, resource)

    def start_collaboration(self):
        print("Initiating Human-AI collaboration...")
        self.environment.collaborate()
