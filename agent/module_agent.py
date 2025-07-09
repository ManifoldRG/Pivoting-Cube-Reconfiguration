from agent.base_agent import Agent
import math
import numpy as np

class ModuleAgent(Agent):
    def __init__(self, module, ogm, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps
        self.steps_taken = 0
        self.success = False
        self.module = module
        self.ogm = ogm
        self.past_v_final_local_map = set() 
        self.curr_v_final_local_map = self.ogm.calc_curr_v_final_local_map(self.module)
        self.past_v_final_local_map = self.curr_v_final_local_map
        

    def calc_self_reward(self):
        return len(self.curr_v_final_local_map - self.past_v_final_local_map) - len(self.past_v_final_local_map - self.curr_v_final_local_map)
    
    def get_messages(self, comms_hub):
        return (comms_hub.get_messages(self, self.module)) # placeholder method for a placeholder comms hub class; it might need more inputs
    
    