import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


# init = {1: (11, 11, 11), 2: (11, 11, 10), 3: (11, 10, 10), 4: (10, 10, 10), 5: (11, 9, 10), 6: (12, 11, 11), 7: (11, 11, 12), 8: (12, 10, 10), 9: (12, 11, 12), 10: (10, 11, 10)}
# final = {1: (4, -1, 12), 2: (4, -1, 13), 3: (4, 0, 12), 4: (4, 0, 13), 5: (4, -1, 11), 6: (4, 0, 11), 7: (3, -1, 12), 8: (3, -1, 11), 9: (2, -1, 11), 10: (5, 0, 11)}
def draw_cube(ax, position, color='blue', alpha=0.5):
    """Draw a unit cube centered at position"""
    # For integer grid, cube corners: from position to position + 1
    x, y, z = position
    # Define vertices of the cube
    r = [0, 1]
    verts = [
        [(x+i, y+j, z+k) for i, j, k in [
            (0,0,0), (1,0,0), (1,1,0), (0,1,0)
        ]],
        [(x+i, y+j, z+k) for i, j, k in [
            (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        ]],
        [(x+i, y+j, z+k) for i, j, k in [
            (0,0,0), (1,0,0), (1,0,1), (0,0,1)
        ]],
        [(x+i, y+j, z+k) for i, j, k in [
            (0,1,0), (1,1,0), (1,1,1), (0,1,1)
        ]],
        [(x+i, y+j, z+k) for i, j, k in [
            (0,0,0), (0,1,0), (0,1,1), (0,0,1)
        ]],
        [(x+i, y+j, z+k) for i, j, k in [
            (1,0,0), (1,1,0), (1,1,1), (1,0,1)
        ]]
    ]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))

def plot(init, final):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot init config as blue cubes
    for pos in init.values():
        draw_cube(ax, pos, color='blue', alpha=0.6)

    # Plot final config as red cubes
    for pos in final.values():
        draw_cube(ax, pos, color='red', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('MSSA Initial (blue) vs Final (red) Configuration')

    # Adjust axes limits
    all_positions = list(init.values()) + list(final.values())
    coords = np.array(all_positions)
    ax.set_xlim(coords[:,0].min() - 1, coords[:,0].max() + 2)
    ax.set_ylim(coords[:,1].min() - 1, coords[:,1].max() + 2)
    ax.set_zlim(coords[:,2].min() - 1, coords[:,2].max() + 2)

    plt.show()


if __name__ == '__main__':

    # Example configurations
    init = {
        1: (11, 13, 2), 2: (11, 14, 2), 3: (11, 15, 2), 4: (10, 13, 2), 5: (10, 15, 2),
        6: (9, 13, 2), 7: (11, 14, 3), 8: (11, 14, 1), 9: (10, 14, 2), 10: (11, 15, 3)
    }

    final = {
        1: (0, 3, 19), 2: (0, 3, 20), 3: (0, 3, 18), 4: (0, 4, 20), 5: (0, 2, 19),
        6: (0, 4, 21), 7: (0, 3, 17), 8: (0, 2, 20), 9: (0, 1, 19), 10: (0, 1, 20)
    }

    plot(init, final)