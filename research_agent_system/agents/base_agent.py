class BaseAgent:
    def __init__(self, name, llm):
        self.name = name
        self.llm = llm

    def run(self, input_data):
        raise NotImplementedError("Each agent must implement run()")
