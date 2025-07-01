# Re-import necessary packages after code execution environment reset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib.animation import PillowWriter

# Re-define StepVisualizer since environment was reset
class StepVisualizer:
    def __init__(self, ogm, output_path="random_search_steps.gif"):
        self.ogm = ogm
        self.output_path = output_path
        self.frames = []
        self.final_state = list(ogm.final_module_positions.values())

    def capture_state(self):
        """Capture current grid state as a list of module positions"""
        module_positions = list(self.ogm.module_positions.values())
        self.frames.append(module_positions.copy())

    def set_final_state(self, final_module_positions):
        self.final_state = list(final_module_positions.values())

    def draw_cube(self, ax, position, color='skyblue', alpha=0.9):
        x, y, z = position
        r = [0, 1]
        vertices = np.array([[x+i, y+j, z+k] for i in r for j in r for k in r])
        faces = [[vertices[j] for j in [0,1,3,2]],
                 [vertices[j] for j in [4,5,7,6]],
                 [vertices[j] for j in [0,1,5,4]],
                 [vertices[j] for j in [2,3,7,6]],
                 [vertices[j] for j in [1,3,7,5]],
                 [vertices[j] for j in [0,2,6,4]]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))

    ### Helpful to visualize in jupyter
    def animate_inline(self):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        def update(frame_idx):
            ax.clear()
            ax.set_xlim(0, self.ogm.grid_map.shape[0])
            ax.set_ylim(0, self.ogm.grid_map.shape[1])
            ax.set_zlim(0, self.ogm.grid_map.shape[2])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Step {frame_idx + 1}")

            if self.final_state:
                for pos in self.final_state:
                    self.draw_cube(ax, pos, color='red', alpha=0.4)
    
            for pos in self.frames[frame_idx]:
                self.draw_cube(ax, pos)
    
        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=500)
        return HTML(ani.to_jshtml())
    
    def animate(self, pause_frames=15):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame_idx):
            ax.clear()
            ax.set_xlim(0, self.ogm.grid_map.shape[0])
            ax.set_ylim(0, self.ogm.grid_map.shape[1])
            ax.set_zlim(0, self.ogm.grid_map.shape[2])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            actual_idx = min(frame_idx, len(self.frames) - 1)
            ax.set_title(f"Step {actual_idx + 1}")

            if self.final_state:
                for pos in self.final_state:
                    self.draw_cube(ax, pos, color='red', alpha=0.4)

            for pos in self.frames[actual_idx]:
                self.draw_cube(ax, pos)

        total_frames = len(self.frames) + pause_frames
        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=0.05)

        writer = PillowWriter(fps=2, metadata={"loop": 0})
        ani.save(self.output_path, writer=writer)
        # ani.save(self.output_path, writer='pillow', loop=0)
        plt.close(fig)
        print(f"[✔] Animation saved to: {self.output_path}")
        return self.output_path