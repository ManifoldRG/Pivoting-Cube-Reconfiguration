"""
Parallel Step Visualizer

Modified visualizer that captures state once per "timestep" (after all agents 
have moved in parallel), rather than after each individual agent action.

Features:
- Batch state capture for parallel movement visualization
- Configurable FPS for speed control (default 4 = 2x normal speed)
- Success-aware final frame pause
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


class ParallelStepVisualizer:
    def __init__(self, ogm, output_path="parallel_steps.gif", fps=4):
        """Initialize the parallel step visualizer.
        
        Args:
            ogm: OccupancyGridMap instance
            output_path: Path to save the GIF
            fps: Frames per second (default 4 = 2x speed vs normal 2 fps)
        """
        self.ogm = ogm
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        self.final_state = list(ogm.final_module_positions.values())

    def capture_parallel_step(self):
        """Capture current grid state after all agents have moved.
        
        Call this once per timestep, after all agents complete their actions.
        """
        module_positions = list(self.ogm.module_positions.values())
        self.frames.append(module_positions.copy())

    def set_final_state(self, final_module_positions):
        """Update the target/final state for visualization."""
        self.final_state = list(final_module_positions.values())

    def draw_cube(self, ax, position, color='skyblue', alpha=0.9):
        """Draw a 3D cube at the given position."""
        x, y, z = position
        r = [0, 1]
        vertices = np.array([[x+i, y+j, z+k] for i in r for j in r for k in r])
        faces = [[vertices[j] for j in [0,1,3,2]],
                 [vertices[j] for j in [4,5,7,6]],
                 [vertices[j] for j in [0,1,5,4]],
                 [vertices[j] for j in [2,3,7,6]],
                 [vertices[j] for j in [1,3,7,5]],
                 [vertices[j] for j in [0,2,6,4]]]
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha
        ))

    def animate(self, pause_frames=20, success=False):
        """Generate and save the animation as a GIF.
        
        Args:
            pause_frames: Number of extra frames to pause on final state
            success: If True, extends pause on final frame to show completion
            
        Returns:
            str: Path to the saved GIF
        """
        if not self.frames:
            print("[!] No frames captured, skipping animation")
            return None
            
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add extra pause frames for success
        if success:
            pause_frames = max(pause_frames, 30)

        def update(frame_idx):
            ax.clear()
            ax.set_xlim(0, self.ogm.grid_map.shape[0])
            ax.set_ylim(0, self.ogm.grid_map.shape[1])
            ax.set_zlim(0, self.ogm.grid_map.shape[2])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            actual_idx = min(frame_idx, len(self.frames) - 1)
            
            # Show timestep (parallel step, not individual action)
            if frame_idx < len(self.frames):
                ax.set_title(f"Timestep {actual_idx + 1}")
            else:
                ax.set_title(f"Timestep {len(self.frames)} (Complete)")

            # Draw target positions in red (semi-transparent)
            if self.final_state:
                for pos in self.final_state:
                    self.draw_cube(ax, pos, color='red', alpha=0.4)

            # Draw current module positions in blue
            for pos in self.frames[actual_idx]:
                self.draw_cube(ax, pos, color='skyblue', alpha=0.9)

        total_frames = len(self.frames) + pause_frames
        
        # Calculate interval from fps (ms per frame)
        interval = 1000.0 / self.fps
        
        ani = animation.FuncAnimation(
            fig, update, frames=total_frames, interval=interval
        )

        writer = PillowWriter(fps=self.fps, metadata={"loop": 0})
        ani.save(self.output_path, writer=writer)
        plt.close(fig)
        
        print(f"[✔] Animation saved to: {self.output_path}")
        print(f"    Frames: {len(self.frames)} timesteps + {pause_frames} pause")
        print(f"    Speed: {self.fps} fps (2x normal)")
        
        return self.output_path

    def get_frame_count(self):
        """Return number of captured frames."""
        return len(self.frames)
