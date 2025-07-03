from MSSA.agent.base_agent import Agent
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
        self.calc_final_coords(ogm)
        
    def get_coords(self, ogm):
        return ogm.module_positions[self.module]
    
    def calc_final_coords(self):
        self.final_coords = []
        for i in range(len(self.ogm.final_rotated_module_positions)):
            temp_pos = self.ogm.final_rotated_module_positions[i]
            self.final_coords.append(temp_pos[self.module])
    
    def get_local_map(self):
        origin = self.get_coords(self.ogm)
        ogm_shape = self.ogm.curr_grid_map.shape
        limits = [origin[0] - 2, origin[0] + 2, origin[1] - 2, origin[1] + 2, origin[2] - 2, origin[2] + 2]

        for i in range(6):
            if limits[i] < 0:
                limits[i] =  0
            elif limits[i] >= ogm_shape(math.floor(i / 2)):
                limits[i] = ogm_shape(math.floor(i / 2)) - 1
        
        return ogm.curr_grid_map[limits[0]:limits[1], limits[2]:limits[3], limits[4]:limits[5]]
    
    def calc_final_local_maps(self):
        self.final_local_maps = []

        for i in range(len(self.ogm.final_grid_maps)):
            #origin = self.get_final_coords(ogm)
            origin = self.final_coords[i]
            ogm_shape = self.ogm.curr_grid_map.shape
            limits = [origin[0] - 2, origin[0] + 2, origin[1] - 2, origin[1] + 2, origin[2] - 2, origin[2] + 2]
            temp_final_map = self.ogm.final_grid_maps[i]

            for i in range(6):
                if limits[i] < 0:
                    limits[i] =  0
                elif limits[i] >= ogm_shape(math.floor(i / 2)):
                    limits[i] = ogm_shape(math.floor(i / 2)) - 1
            self.final_local_maps.append(temp_final_map[limits[0]:limits[1], limits[2]:limits[3], limits[4]:limits[5]])
        
        #return ogm.final_grid_map[limits[0]:limits[1], limits[2]:limits[3], limits[4]:limits[5]]
    
    def calc_curr_v_final_local_map(self):
        local_map = self.get_local_map()
        min_set = set()

        for i in range(len(self.final_local_maps)):
            temp_mask = local_map == self.final_local_maps[i]
            temp_set = set(local_map[temp_mask].flatten())
            
            if len(temp_set) < len(min_set):
                min_set = temp_set

        return min_set
    
    def get_messages(self, comms_hub):
        return (comms_hub.get_messages(self, self.module)) # placeholder method for a placeholder comms hub class; it might need more inputs
