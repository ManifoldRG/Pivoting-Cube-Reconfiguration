import numpy as np
from env.pivoting_cubes_env import PivotingCubesEnv

initial_positions = {
    1: (3, 3, 3),
    2: (3, 4, 3)
}
final_positions = {
    1: (3, 3, 3),
    2: (4, 3, 3)
}
n_modules = 2

def main():
    env = PivotingCubesEnv(initial_positions, final_positions, n_modules, empathy_lambda=0.1, max_episode_steps=10)
    obs, infos = env.reset()
    print("Initial observations:")
    for agent, ob in obs.items():
        print(f"{agent}: {ob}")
    done = False
    step = 0
    while not done and step < 10:
        actions = {}
        for agent in env.agents:
            mask = obs[agent]["action_mask"]
            legal_actions = np.where(mask)[0]
            actions[agent] = np.random.choice(legal_actions)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        print(f"\nStep {step+1}")
        print("Actions:", actions)
        print("Rewards:", rewards)
        print("Terminations:", terminations)
        print("Truncations:", truncations)
        done = not env.agents or all(terminations.values()) or all(truncations.values())
        step += 1
    print("\nFinal module positions:")
    env.render()

if __name__ == "__main__":
    main() 