import logging
import numpy as np
## Need to generate random configuration 
## Model it as a 3D graph with 1 connected component
## So pick a random node and then pick random direction to make the modules

def make_connected_positions(n, grid_size, starter = None):
    directions = [
        np.array([1, 0, 0]), 
        np.array([-1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, -1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, -1])
    ]

    positions = {}
    used = set()

    root = tuple(np.random.randint(0, grid_size, size=3))
    if starter is not None:
        root = starter

    positions[1] = root


    used.add(root)

    for i in range(2, n+1):
        while True:
            found = False
            parent_id = np.random.choice(list(positions.keys()))
            parent_pos = np.array(positions[parent_id])



            np.random.shuffle(directions)
            for dir in directions:
                candidate = tuple(parent_pos + dir)
                if all(0 <= c < grid_size for c in candidate) and candidate not in used:
                    positions[i] = candidate
                    used.add(candidate)
                    found = True
                    break
            if found:
                break

    logging.debug("Generated connected positions for %s modules", n)
    logging.debug(positions)
    return positions

    # Start root at 


def random_configuration(n):
    # Can change grid size matching with ogm, but recentering might cause problem, so made n + 3 instead of 2n+3
    grid_size = max(5,  n + 3)

    init = make_connected_positions(n, grid_size)
    final = make_connected_positions(n, grid_size, init[np.random.choice(list(init.keys()))])
    return init, final, max(5, 2*n+3)