"""
Demo Configuration Module

Easily editable module coordinates for fixed initial/final configurations.
All coordinates are (x, y, z) tuples on a 3D grid.

To modify configurations:
1. Edit the coordinate dictionaries below
2. Ensure all modules are connected (face-adjacent) 
3. Keep module 1 at or near center for proper recentering
"""

# ==============================================================================
# PLUS-SIGN SHAPE (Initial Configuration)
# ==============================================================================
# Visual representation (XY plane at z=4):
#   OXO
#   XXX
#   OXO
#
# Where X = module, O = empty
# Module 1 is at center for recentering stability

C_SHAPE_COORDS = {
    1: (4, 4, 4),  # Center
    2: (4, 5, 4),  # Top (+Y)
    3: (4, 3, 4),  # Bottom (-Y)
    4: (4, 2, 4),  # Right (+X)
    5: (4, 6, 4),  # Left (-X)
}

# ==============================================================================
# C-SHAPE (Final Configuration)
# ==============================================================================
# Visual representation (XY plane at z=4):
#   XXO
#   XOO
#   XXO
#
# Vertical bar on left with arms extending right at top and bottom
# Module 1 stays at center-left for matching

PLUS_SHAPE_COORDS = {
    1: (4, 4, 4),  # Center-left (vertical bar middle)
    2: (3, 4, 4),  # Top-left (vertical bar top)
    3: (2, 4, 4),  # Bottom-left (vertical bar bottom)
    4: (5, 4, 4),  # Top-right arm
    5: (6, 4, 4),  # Bottom-right arm
}

# ==============================================================================
# CONFIGURATION METADATA
# ==============================================================================

NUM_AGENTS = 5
GRID_SIZE = max(5, NUM_AGENTS * 2 + 3)  # Consistent with OGM sizing


def get_configs():
    """Return the initial and final configurations.
    
    Returns:
        tuple: (initial_config, final_config, grid_size)
    """
    return PLUS_SHAPE_COORDS.copy(), C_SHAPE_COORDS.copy(), GRID_SIZE


def validate_connectivity(config):
    """Check if all modules in config are connected (face-adjacent).
    
    Args:
        config: Dictionary mapping module IDs to (x, y, z) positions
        
    Returns:
        bool: True if configuration is fully connected
    """
    if len(config) <= 1:
        return True
    
    positions = set(config.values())
    visited = set()
    start = next(iter(positions))
    stack = [start]
    
    while stack:
        pos = stack.pop()
        if pos in visited:
            continue
        visited.add(pos)
        
        # Check 6 face-adjacent neighbors
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            neighbor = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            if neighbor in positions and neighbor not in visited:
                stack.append(neighbor)
    
    return len(visited) == len(positions)


if __name__ == "__main__":
    # Quick validation when run directly
    print("Plus Shape Configuration:")
    for mod, pos in PLUS_SHAPE_COORDS.items():
        print(f"  Module {mod}: {pos}")
    print(f"  Connected: {validate_connectivity(PLUS_SHAPE_COORDS)}")
    
    print("\nC-Shape Configuration:")
    for mod, pos in C_SHAPE_COORDS.items():
        print(f"  Module {mod}: {pos}")
    print(f"  Connected: {validate_connectivity(C_SHAPE_COORDS)}")
