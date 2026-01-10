from agent.base_agent import Agent
import numpy as np
from ogm.ogm_env import OGMEnv
class RandomSearchAgent(Agent):
    def __init__(self, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps
        self.steps_taken = 0
        self.success = False
    def select_action(self, available_actions, num_modules):
        actions_to_take = {}

        for m in range(1,num_modules+1):
            actions_to_take[m] = np.where(available_actions[m])[0] + 1

        module = np.random.randint(1,m+1)
        actions = actions_to_take[module]

        while len(actions) < 1:
            module = np.random.randint(1,m+1)
            actions = actions_to_take[module]

        return (module, actions[np.random.randint(len(actions))])

    def search(self, ogm: OGMEnv):

        while self.steps_taken < self.max_steps:
            possible_actions = ogm.ogm.calc_possible_actions()
            module, action = self.select_action(possible_actions, ogm.num_modules)
            observation,reward,done,info  = ogm.step((module,action))
            self.steps_taken += 1
            if done:
                if info["is_success"]:
                    self.success = True
                    print(f"Goal reached in {self.steps_taken} steps!")
                    return True
                break

        print(f"Failed to reach goal in {self.steps_taken} steps.")
        return False
