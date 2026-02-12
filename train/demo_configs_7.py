"""
7-Agent Demo Configuration Module

Easily editable module coordinates for 7-agent fixed initial/final configurations.
All coordinates are (x, y, z) tuples on a 3D grid.

To modify configurations:
1. Edit the coordinate dictionaries below
2. Ensure all modules are connected (face-adjacent) 
3. Keep module 1 at or near center for proper recentering
"""

# ==============================================================================
# INITIAL CONFIGURATION: 3D Plus/Cross (7 agents)
# ==============================================================================
# A 3D plus with:
# - Center cube (module 1)
# - 4 arms in XY plane (like 5-agent plus)
# - 1 cube in front (+Z) and 1 cube behind (-Z) the center
#
# Visual: XY plane at z=4 shows the cross, plus z=3 and z=5 have center cubes

INITIAL_COORDS_7 = {
    1: (4, 4, 4),  # Center
    2: (4, 5, 4),  # +Y (top in XY plane)
    3: (4, 3, 4),  # -Y (bottom in XY plane)
    4: (5, 4, 4),  # +X (right in XY plane)
    5: (3, 4, 4),  # -X (left in XY plane)
    6: (4, 4, 5),  # +Z (front)
    7: (4, 4, 3),  # -Z (back)
}

# ==============================================================================
# FINAL CONFIGURATION: C-Shape with 7 agents
# ==============================================================================
# Visual representation (XY plane at z=4):
#   X X X
#   X O O
#   X X X
#
# Horizontal bars at top and bottom, connected on left

FINAL_COORDS_7 = {
    1: (4, 4, 4),  # Left middle (connector)
    2: (4, 5, 4),  # Top-left
    3: (4, 3, 4),  # Bottom-left
    4: (3, 5, 4),  # Top-middle
    5: (2, 5, 4),  # Top-right
    6: (3, 3, 4),  # Bottom-middle
    7: (2, 3, 4),  # Bottom-right
}

# ==============================================================================
# CONFIGURATION METADATA
# ==============================================================================

NUM_AGENTS = 7
GRID_SIZE = max(5, NUM_AGENTS * 2 + 3)  # Consistent with OGM sizing


def get_configs():
    """Return the initial and final configurations.
    
    Returns:
        tuple: (initial_config, final_config, grid_size)
    """
    return INITIAL_COORDS_7.copy(), FINAL_COORDS_7.copy(), GRID_SIZE


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
    print("Initial Configuration (Line):")
    for mod, pos in INITIAL_COORDS_7.items():
        print(f"  Module {mod}: {pos}")
    print(f"  Connected: {validate_connectivity(INITIAL_COORDS_7)}")
    
    print("\nFinal Configuration (U-Shape):")
    for mod, pos in FINAL_COORDS_7.items():
        print(f"  Module {mod}: {pos}")
    print(f"  Connected: {validate_connectivity(FINAL_COORDS_7)}")
