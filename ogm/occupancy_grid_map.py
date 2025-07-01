import numpy as np

class OccupancyGridMap:
  def __init__(self, module_positions, final_module_positions, n):
    """Initialize the occupancy grid map with module positions.
    
    Args:
        module_positions: Dictionary mapping module numbers to their positions (x,y,z)
        final_module_positions: Dictionary mapping module numbers to their goal positions (x,y,z)
        n: Number of modules
    """
    # Validate inputs
    if not module_positions or not final_module_positions:
        raise ValueError("Module positions dictionaries cannot be empty")
    if n <= 0:
        raise ValueError("Number of modules must be positive")
    
    # Store original module positions before recentering
    self.original_module_positions = module_positions.copy()
    self.original_final_module_positions = final_module_positions.copy()
    
    # Calculate grid size based on number of modules
    grid_size = self.calculate_grid_size(n)
    
    # Create grid maps with appropriate size
    self.grid_map = np.zeros((grid_size, grid_size, grid_size))
    self.curr_grid_map = np.zeros((grid_size, grid_size, grid_size))
    self.final_grid_map = np.zeros((grid_size, grid_size, grid_size))
    
    # Recenter module positions so that module 1 is at the center of the grid
    self.module_positions, self.final_module_positions = self.recenter_initial_positions(
        module_positions, final_module_positions, grid_size)
    

    # Initialize grid maps with recentered module positions
    for module, pos in self.module_positions.items():
        self.grid_map[pos[0], pos[1], pos[2]] = module
        self.curr_grid_map[pos[0], pos[1], pos[2]] = module
    
    for module, pos in self.final_module_positions.items():
        self.final_grid_map[pos[0], pos[1], pos[2]] = module
    
    # Set reference position for recentering during operations
    self.recenter_to = self.module_positions[1]
    self.modules = range(1, n+1)
    self.edges = self.calculate_edges(self.modules, self.module_positions)
    self.rotation_matrices()
    self.init_actions()

  def calculate_grid_size(self, n):
    """Calculate grid size based on number of modules.
    
    Args:
        n: Number of modules
        
    Returns:
        Grid size (same for all dimensions)
    """
    # Ensure minimum grid size of 5x5x5
    # For larger module counts, use a formula that scales with module count
    # Using n*2+3 as a simple scaling formula
    return max(5, n*2+3)
  
  def recenter_initial_positions(self, module_positions, final_module_positions, grid_size):
    """Recenter module positions so that module 1 is at the center of the grid.
    
    Args:
        module_positions: Original module positions dictionary
        final_module_positions: Original final module positions dictionary
        grid_size: Size of the grid
        
    Returns:
        Tuple of (recentered module positions, recentered final module positions)
    """
    # Calculate center of the grid
    grid_center = grid_size // 2
    
    # Get position of module 1
    if 1 not in module_positions:
        raise ValueError("Module 1 must exist in the module positions dictionary")
    
    module1_pos = module_positions[1]
    
    # Calculate offset to move module 1 to grid center
    offset = (
        grid_center - module1_pos[0],
        grid_center - module1_pos[1],
        grid_center - module1_pos[2]
    )
    
    # Apply offset to all module positions
    recentered_positions = {}
    for module, pos in module_positions.items():
        recentered_positions[module] = (
            pos[0] + offset[0],
            pos[1] + offset[1],
            pos[2] + offset[2]
        )
    
    # Apply offset to all final module positions
    recentered_final_positions = {}
    for module, pos in final_module_positions.items():
        recentered_final_positions[module] = (
            pos[0] + offset[0],
            pos[1] + offset[1],
            pos[2] + offset[2]
        )
    
    return recentered_positions, recentered_final_positions

  # recenter the grid_map so that a module (the first one for now) is at (0,0,0)
  def recenter(self):
    # recenter to a position (NOT the origin)
    #ipdb.set_trace()
    curr_pos = self.module_positions[1]
    offset = (curr_pos[0] - self.recenter_to[0], curr_pos[1] - self.recenter_to[1], curr_pos[2] - self.recenter_to[2])
    self.curr_grid_map = np.zeros(self.curr_grid_map.shape)

    for module in self.modules:
      temp_mod = self.module_positions[module]
      new_pos = (temp_mod[0] - offset[0], temp_mod[1] - offset[1], temp_mod[2] - offset[2])
      self.module_positions[module] = new_pos
      self.curr_grid_map[new_pos[0], new_pos[1], new_pos[2]] = module


  # probably need each module to track its own position so that they can be easily recentered

  # define possible actions in terms of a dictionary of a list of vectors from module to other necessary modules and a list of vectors from module to necessary empty spaces for keys, and values are vectors where the module ends up.
  # actually use the slices as in the problem formulation. possible_actions is a dictionary with modules as keys and a 48 boolean long list as values.
  def init_actions(self):
    self.potential_pivots = {1: np.array([[True, True, False], [True, False, False]]), #fixed
                             2: np.array([[True, True, False], [False, False, False], [False, False, False]]), #fixed
                             3: np.array([[False, True, True], [False, False, True]]), #fixed
                             4: np.array([[False, True, True], [False, False, False], [False, False, False]]), #fixed
                             5: np.array([[False, False], [True, False], [True, True]]), #fixed
                             6: np.array([[False, False, False], [True, False, False], [True, False, False]]), #fixed
                             7: np.array([[False, False], [False, True], [True, True]]), #fixed
                             8: np.array([[False, False, False], [False, False, True], [False, False, True]]), #fixed
                             9: np.array([[True, False, False], [True, True, False]]), #fixed
                             10: np.array([[False, False, False], [False, False, False], [True, True, False]]), #fixed
                             11: np.array([[False, False, True], [False, True, True]]), #fixed
                             12: np.array([[False, False, False], [False, False, False], [False, True, True]]), #fixed
                             13: np.array([[True, True], [True, False], [False, False]]), #fixed
                             14: np.array([[True, False, False], [True, False, False], [False, False, False]]), #fixed
                             15: np.array([[True, True], [False, True], [False, False]]), #fixed
                             16: np.array([[False, False, True], [False, False, True], [False, False, False]]), #fixed
                             17: np.array([[True, True, False], [True, False, False]]), #fixed
                             18: np.array([[True, True, False], [False, False, False], [False, False, False]]), #fixed
                             19: np.array([[False, True, True], [False, False, True]]), #fixed
                             20: np.array([[False, True, True], [False, False, False], [False, False, False]]), #fixed
                             21: np.array([[False, False], [True, False], [True, True]]), #fixed
                             22: np.array([[False, False, False], [True, False, False], [True, False, False]]), #fixed
                             23: np.array([[False, False], [False, True], [True, True]]), #fixed
                             24: np.array([[False, False, False], [False, False, True], [False, False, True]]), #fixed
                             25: np.array([[True, False, False], [True, True, False]]), #fixed
                             26: np.array([[False, False, False], [False, False, False], [True, True, False]]), #fixed
                             27: np.array([[False, False, True], [False, True, True]]), #fixed
                             28: np.array([[False, False, False], [False, False, False], [False, True, True]]), #fixed
                             29: np.array([[True, True], [True, False], [False, False]]), #fixed
                             30: np.array([[True, False, False], [True, False, False], [False, False, False]]), #fixed
                             31: np.array([[True, True], [False, True], [False, False]]), #fixed
                             32: np.array([[False, False, True], [False, False, True], [False, False, False]]), #fixed
                             33: np.array([[True, True, False], [True, False, False]]), #fixed
                             34: np.array([[True, True, False], [False, False, False], [False, False, False]]), #fixed
                             35: np.array([[False, True, True], [False, False, True]]), #fixed
                             36: np.array([[False, True, True], [False, False, False], [False, False, False]]), #fixed
                             37: np.array([[False, False], [True, False], [True, True]]), #fixed
                             38: np.array([[False, False, False], [True, False, False], [True, False, False]]), #fixed
                             39: np.array([[False, False], [False, True], [True, True]]), #fixed
                             40: np.array([[False, False, False], [False, False, True], [False, False, True]]), #fixed
                             41: np.array([[True, False, False], [True, True, False]]), #fixed
                             42: np.array([[False, False, False], [False, False, False], [True, True, False]]), #fixed
                             43: np.array([[False, False, True], [False, True, True]]), #fixed
                             44: np.array([[False, False, False], [False, False, False], [False, True, True]]), #fixed
                             45: np.array([[True, True], [True, False], [False, False]]), #fixed
                             46: np.array([[True, False, False], [True, False, False], [False, False, False]]), #fixed
                             47: np.array([[True, True], [False, True], [False, False]]), #fixed
                             48: np.array([[False, False, True], [False, False, True], [False, False, False]]) #fixed
                             }

    # 3 rows for x, y, z, respectively, with start, stop
    self.ranges = {1: np.array([[0,1], [-1,1], [0,0]]),
                   2: np.array([[0,2], [-1,1], [0,0]]),
                   3: np.array([[0,1], [-1,1], [0,0]]),
                   4: np.array([[0,2], [-1,1], [0,0]]),
                   5: np.array([[-1,1], [0,1], [0,0]]),
                   6: np.array([[-1,1], [0,2], [0,0]]),
                   7: np.array([[-1,1], [-1,0], [0,0]]), # does the negative stuff work?
                   8: np.array([[-1,1], [-2,0], [0,0]]), # does the negative stuff work?
                   9: np.array([[-1,0], [-1,1], [0,0]]),
                   10: np.array([[-2,0], [-1,1], [0,0]]),
                   11: np.array([[-1,0], [-1,1], [0,0]]),
                   12: np.array([[-2,0], [-1,1], [0,0]]),
                   13: np.array([[-1,1], [0,1], [0,0]]),
                   14: np.array([[-1,1], [0,2], [0,0]]),
                   15: np.array([[-1,1], [-1,0], [0,0]]), # does the negative stuff work?
                   16: np.array([[-1,1], [-2,0], [0,0]]), # does the negative stuff work? # now switch which dimension stays the same
                   17: np.array([[0,1], [0,0], [-1,1]]),
                   18: np.array([[0,2], [0,0], [-1,1]]),
                   19: np.array([[0,1], [0,0], [-1,1]]),
                   20: np.array([[0,2], [0,0], [-1,1]]),
                   21: np.array([[-1,1], [0,0], [0,1]]),
                   22: np.array([[-1,1], [0,0], [0,2]]),
                   23: np.array([[-1,1], [0,0], [-1,0]]), # does the negative stuff work?
                   24: np.array([[-1,1], [0,0], [-2,0]]), # does the negative stuff work?
                   25: np.array([[-1,0], [0,0], [-1,1]]),
                   26: np.array([[-2,0], [0,0], [-1,1]]),
                   27: np.array([[-1,0], [0,0], [-1,1]]),
                   28: np.array([[-2,0], [0,0], [-1,1]]),
                   29: np.array([[-1,1], [0,0], [0,1]]),
                   30: np.array([[-1,1], [0,0], [0,2]]),
                   31: np.array([[-1,1], [0,0], [-1,0]]), # does the negative stuff work?
                   32: np.array([[-1,1], [0,0], [-2,0]]), # does the negative stuff work? # now switch which dimension stays the same
                   33: np.array([[0,0], [0,1], [-1,1]]),
                   34: np.array([[0,0], [0,2], [-1,1]]),
                   35: np.array([[0,0], [0,1], [-1,1]]),
                   36: np.array([[0,0], [0,2], [-1,1]]),
                   37: np.array([[0,0], [-1,1], [0,1]]),
                   38: np.array([[0,0], [-1,1], [0,2]]),
                   39: np.array([[0,0], [-1,1], [-1,0]]), # does the negative stuff work?
                   40: np.array([[0,0], [-1,1], [-2,0]]), # does the negative stuff work?
                   41: np.array([[0,0], [-1,0], [-1,1]]),
                   42: np.array([[0,0], [-2,0], [-1,1]]),
                   43: np.array([[0,0], [-1,0], [-1,1]]),
                   44: np.array([[0,0], [-2,0], [-1,1]]),
                   45: np.array([[0,0], [-1,1], [0,1]]),
                   46: np.array([[0,0], [-1,1], [0,2]]),
                   47: np.array([[0,0], [-1,1], [-1,0]]), # does the negative stuff work?
                   48: np.array([[0,0], [-1,1], [-2,0]]) # does the negative stuff work? # now switch which dimension stays the same
                   }

    self.negative_y_ranges = {7,8,15,16}
    self.negative_y_ranges = {}
    self.negative_z_ranges = {23,24,31,32,39,40,47,48}
    self.negative_z_ranges = {}

  def calc_possible_actions(self): # need to check now that neighbor is free
    self.possible_actions = {}
    self.articulation_points = set(self.articulationPoints(len(self.modules), self.edges))
    # print("articulation_points\n")
    # print(self.articulation_points)

    for m in self.modules:
      #ipdb.set_trace()
      self.possible_actions[m] = np.array(list(range(48))) > 49

      if m not in self.articulation_points:
        module_position = self.module_positions[m]

        # will go to 48
        for p in range(1, 49):
          #ipdb.set_trace()
          rangethingy = self.ranges[p]
          offset_x = module_position[0] + rangethingy[0]
          offset_y = module_position[1] + rangethingy[1]
          offset_z = module_position[2] + rangethingy[2]

          if p in self.negative_y_ranges:
            sliced = self.curr_grid_map[offset_x[0]:(offset_x[1] + 1), offset_y[0]:(None if offset_y[1]==0 else offset_y[1] - 1):-1, offset_z[0]:(offset_z[1] + 1)]
          elif p in self.negative_z_ranges:
            sliced = self.curr_grid_map[offset_x[0]:(offset_x[1] + 1), offset_y[0]:(offset_y[1] + 1), offset_z[0]:(None if offset_z[1]==0 else offset_z[1] - 1):-1]
          else:
            sliced = self.curr_grid_map[offset_x[0]:(offset_x[1] + 1), offset_y[0]:(offset_y[1] + 1), offset_z[0]:(offset_z[1] + 1)]

          booled = np.squeeze(sliced > 0)
          pa = self.possible_actions[m]
          pa[p - 1] = np.all(booled == self.potential_pivots[p])
          self.possible_actions[m] = pa
          #print(p)
          #ipdb.set_trace()
    # print(f"Possible actions: ")
    #print(self.possible_actions)

    # for m in self.modules:
    #   print(np.where(self.possible_actions[m])[0] + 1)

    return self.possible_actions

  def take_action(self, module, action):
    module_position = self.module_positions[module]

    match action:
      case 1:
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2])
      case 2:
        new_module_position = (module_position[0] + 1, module_position[1] - 1, module_position[2])
      case 3:#
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2])
      case 4:
        new_module_position = (module_position[0] + 1, module_position[1] + 1, module_position[2])
      case 5:
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2])
      case 6:
        new_module_position = (module_position[0] + 1, module_position[1] + 1, module_position[2])
      case 7:
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2])
      case 8:
        new_module_position = (module_position[0] + 1, module_position[1] - 1, module_position[2])
      case 9:
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2])
      case 10:
        new_module_position = (module_position[0] - 1, module_position[1] - 1, module_position[2])
      case 11:
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2])
      case 12:#
        new_module_position = (module_position[0] - 1, module_position[1] + 1, module_position[2])
      case 13:
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2])
      case 14:
        new_module_position = (module_position[0] - 1, module_position[1] + 1, module_position[2])
      case 15:
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2])
      case 16:#####################
        new_module_position = (module_position[0] - 1, module_position[1] - 1, module_position[2])
      case 17:
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2])
      case 18:
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2] - 1) #fixed
      case 19:#
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2])
      case 20:
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2] + 1) #fixed
      case 21:
        new_module_position = (module_position[0], module_position[1], module_position[2] + 1) #fixed
      case 22:
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2] + 1) #fixed
      case 23:
        new_module_position = (module_position[0], module_position[1], module_position[2] - 1) #fixed
      case 24:
        new_module_position = (module_position[0] + 1, module_position[1], module_position[2] - 1)
      case 25:
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2])
      case 26:
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2] - 1)
      case 27:
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2])
      case 28:#
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2] + 1)
      case 29:
        new_module_position = (module_position[0], module_position[1], module_position[2] + 1)
      case 30:
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2] + 1)
      case 31:
        new_module_position = (module_position[0], module_position[1], module_position[2] - 1)
      case 32:#####################
        new_module_position = (module_position[0] - 1, module_position[1], module_position[2] - 1)
      case 33:
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2])
      case 34:
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2] - 1)
      case 35:#
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2])
      case 36:
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2] + 1)
      case 37:
        new_module_position = (module_position[0], module_position[1], module_position[2] + 1)
      case 38:
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2] + 1) 
      case 39:
        new_module_position = (module_position[0], module_position[1], module_position[2] - 1)
      case 40:
        new_module_position = (module_position[0], module_position[1] + 1, module_position[2] - 1)
      case 41:
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2])
      case 42:
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2] - 1)
      case 43:
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2])
      case 44:#
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2] + 1)
      case 45:
        new_module_position = (module_position[0], module_position[1], module_position[2] + 1)
      case 46:
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2] + 1)
      case 47:
        new_module_position = (module_position[0], module_position[1], module_position[2] - 1)
      case 48:#####################
        new_module_position = (module_position[0], module_position[1] - 1, module_position[2] - 1)


    self.curr_grid_map[module_position[0], module_position[1], module_position[2]] = 0
    self.curr_grid_map[new_module_position[0], new_module_position[1], new_module_position[2]] = module
    self.module_positions[module] =new_module_position
    self.recenter()
    self.edges = self.calculate_edges(self.modules, self.module_positions)
    
    # print(f"Module Positions: {self.module_positions}")
    #print(f"Curr Grid Map: {self.curr_grid_map}")

  def rotation_matrices(self):
    rx1 = np.array([[1, 0, 0], [0, np.cos(np.pi / 2), -np.sin(np.pi / 2)], [0, np.sin(np.pi / 2), np.cos(np.pi / 2)]])
    rx2 = np.array([[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]])
    rx3 = np.array([[1, 0, 0], [0, np.cos(3 * np.pi / 2), -np.sin(3 * np.pi / 2)], [0, np.sin(3 * np.pi / 2), np.cos(3 * np.pi / 2)]])
    ry1 = np.array([[np.cos(np.pi / 2), 0, np.sin(np.pi / 2)], [0, 1, 0], [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]])
    ry2 = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
    ry3 = np.array([[np.cos(3 * np.pi / 2), 0, np.sin(3 * np.pi / 2)], [0, 1, 0], [-np.sin(3 * np.pi / 2), 0, np.cos(3 * np.pi / 2)]])
    rz1 = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0], [np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 1]])
    rz2 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0], [np.sin(np.pi), np.cos(np.pi), 0], [0, 0, 1]])
    rz3 = np.array([[np.cos(3 * np.pi / 2), -np.sin(3 * np.pi / 2), 0], [np.sin(3 * np.pi / 2), np.cos(3 * np.pi / 2), 0], [0, 0, 1]])

    self.rotmats = [rx1, rx2, rx3, ry1, ry2, ry3, rz1, rz2, rz3]
    self.final_grid_maps = [self.final_grid_map]

    for i in range(9):
      temp_grid_map = np.zeros(self.curr_grid_map.shape)
      rotmat = self.rotmats[i]

      for m in self.modules:
        temp_pos = self.final_module_positions[m]
        temp_pos = np.subtract(temp_pos, self.recenter_to)
        new_pos = np.matmul(rotmat, temp_pos)
        new_pos = np.rint(new_pos)
        new_pos = new_pos.astype(int)
        new_pos = np.add(new_pos, self.recenter_to)
        #ipdb.set_trace()
        # print(new_pos)
        temp_grid_map[new_pos[0], new_pos[1], new_pos[2]] = m

      self.final_grid_maps.append(temp_grid_map)
    # print(f"Final grid maps: {self.final_grid_maps}")

  def check_final(self):
    #ipdb.set_trace()
    for i in range(len(self.final_grid_maps)):
      if np.all(self.curr_grid_map == self.final_grid_maps[i]):
        return True
    return False
    #return np.all(self.curr_grid_map == self.final_grid_map)
  # need to check relative positions of modules, maybe with a connectivity graph

  # need to calculate edges first
  def calculate_edges(self, modules, module_positions):
    edges = []

    for m in modules:
      for n in range(m + 1, len(modules) + 1):
        pos_m = module_positions[m]
        pos_n = module_positions[n]

        if np.sum(np.abs(np.subtract(pos_m, pos_n))) == 1:
          edges.append([m-1,n-1])

    # print("edges:")
    # print(edges)
    return edges


  def constructAdj(self, V, edges):
      adj = [[] for _ in range(V)]

      for edge in edges:
          adj[edge[0]].append(edge[1])
          adj[edge[1]].append(edge[0])
      return adj

  # Helper function to perform DFS and find articulation points
  # using Tarjan's algorithm.
  def findPoints(self, adj, u, visited, disc, low, time, parent, isAP):

      # Mark vertex u as visited and assign discovery
      # time and low value
      visited[u] = 1
      time[0] += 1
      disc[u] = low[u] = time[0]
      children = 0

      # Process all adjacent vertices of u
      for v in adj[u]:

          # If v is not visited, then recursively visit it
          if not visited[v]:
              children += 1
              self.findPoints(adj, v, visited, disc, low, time, u, isAP)

              # Check if the subtree rooted at v has a
              # connection to one of the ancestors of u
              low[u] = min(low[u], low[v])

              # If u is not a root and low[v] is greater than or equal to disc[u],
              # then u is an articulation point
              if parent != -1 and low[v] >= disc[u]:
                  isAP[u] = 1

          # Update low value of u for back edge
          elif v != parent:
              low[u] = min(low[u], disc[v])

      # If u is root of DFS tree and has more than
      # one child, it is an articulation point
      if parent == -1 and children > 1:
          isAP[u] = 1

  # Main function to find articulation points in the graph
  def articulationPoints(self, V, edges):

      #ipdb.set_trace()
      adj = self.constructAdj(V, edges)
      # print("adjacency:")
      # print(adj)
      disc = [0] * V
      low = [0] * V
      visited = [0] * V
      isAP = [0] * V
      time = [0]

      # Run DFS from each vertex if not
      # already visited (to handle disconnected graphs)
      for u in range(V):
          if not visited[u]:
              self.findPoints(adj, u, visited, disc, low, time, -1, isAP)

      # Collect all vertices that are articulation points
      result = [u for u in range(V) if isAP[u]]
      result = [x+1 for x in result]

      # If no articulation points are found, return list containing -1
      return result if result else [-1]
