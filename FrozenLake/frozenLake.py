import json, time, numpy as np, gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
try:
    import pygame
except:
    pygame = None

# ==================== CẤU HÌNH ====================
ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False
MAP_SIZE = 12
OUTPUT_FILE = "best_path.json"
INPUT_FILE = "best_path.json"
ENABLE_UI = True
STEP_DELAY = 0.12
# =================================================

def value_iteration(env, gamma=0.99, tolerance=1e-10, max_iterations=10000):
    base_env = env.unwrapped
    num_states, num_actions = base_env.observation_space.n, base_env.action_space.n
    transitions = base_env.P
    state_values = np.zeros(num_states)

    for _ in range(max_iterations):
        delta = 0
        for state in range(num_states):
            action_values = np.zeros(num_actions)
            for action in range(num_actions):
                for prob, next_state, reward, terminal in transitions[state][action]:
                    action_values[action] += prob * (reward + gamma * (0 if terminal else state_values[next_state]))
            new_value = action_values.max()
            delta = max(delta, abs(new_value - state_values[state]))
            state_values[state] = new_value
        if delta < tolerance:
            break

    policy = np.zeros(num_states, int)
    for state in range(num_states):
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, terminal in transitions[state][action]:
                action_values[action] += prob * (reward + gamma * (0 if terminal else state_values[next_state]))
        policy[state] = action_values.argmax()
    return policy


def create_environment(desc=None):
    if desc is None:
        desc = generate_random_map(size=MAP_SIZE)
    return gym.make(ENV_ID, is_slippery=IS_SLIPPERY, desc=desc)


def run_episode(env, policy, render_callback=None, delay=0.15, max_steps=2000):
    state, _ = env.reset()
    path = [{"state": int(state), "action": None}]
    total_reward = 0.0

    if render_callback:
        render_callback(int(state))
        time.sleep(delay)

    for _ in range(max_steps):
        action = int(policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        path[-1]["action"] = action
        path.append({"state": int(next_state), "action": None})
        state = next_state

        if render_callback:
            render_callback(int(state))
            time.sleep(delay)

        if terminated or truncated:
            break

    return path, total_reward, total_reward > 0


def convert_desc_to_str(desc_array):
    return ["".join((cell.decode() if isinstance(cell, (bytes, np.bytes_)) else str(cell)) for cell in row)
            for row in desc_array.tolist()]


def train_agent():
    print("Bắt đầu huấn luyện")
    env = create_environment()
    policy = value_iteration(env)
    map_description = convert_desc_to_str(env.unwrapped.desc)
    best_path, total_reward, success = run_episode(env, policy)

    json.dump({
        "env_id": ENV_ID,
        "is_slippery": IS_SLIPPERY,
        "size": len(map_description),
        "policy": policy.tolist(),
        "path": best_path,
        "reward": total_reward,
        "success": success,
        "desc": map_description
    }, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False)

    env.close()
    print("Hoàn thành huấn luyện")


def init_ui(desc, scale=48):
    if pygame is None:
        raise RuntimeError("Cài pygame: pip install pygame")
    pygame.init()
    grid_size = len(desc)
    screen = pygame.display.set_mode((grid_size * scale, grid_size * scale))
    pygame.display.set_caption(f"FrozenLake {grid_size}x{grid_size}")
    clock = pygame.time.Clock()
    return screen, clock, scale


def draw_map(screen, desc, agent_state, scale):
    grid_size = len(desc)
    screen.fill((240, 240, 240))
    for row in range(grid_size):
        for col in range(grid_size):
            tile = desc[row][col]
            color = (220, 240, 255)
            if tile == "S":
                color = (200, 255, 200)
            elif tile == "H":
                color = (60, 60, 60)
            elif tile == "G":
                color = (255, 230, 180)
            x, y = col * scale, row * scale
            pygame.draw.rect(screen, color, (x, y, scale, scale))
            pygame.draw.rect(screen, (200, 200, 200), (x, y, scale, scale), 1)
    row, col = divmod(agent_state, grid_size)
    agent_x, agent_y = col * scale + scale // 2, row * scale + scale // 2
    pygame.draw.circle(screen, (30, 144, 255), (agent_x, agent_y), int(scale * 0.32))
    pygame.display.flip()


def test_agent():
    print("Bắt đầu kiểm thử")
    data = json.load(open(INPUT_FILE, "r", encoding="utf-8"))
    desc = data["desc"]
    env = create_environment(desc=desc)
    policy = np.array(data["policy"], int)

    render_callback = None
    if ENABLE_UI:
        screen, clock, scale = init_ui(desc, 48)
        running = True

        def render_callback(state):
            nonlocal running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if running:
                draw_map(screen, desc, state, scale)
                clock.tick(60)

    run_episode(env, policy, render_callback=render_callback, delay=STEP_DELAY)
    if ENABLE_UI and pygame:
        time.sleep(0.3)
        pygame.quit()
    env.close()


if __name__ == "__main__":
    train_agent()
    if ENABLE_UI:
        test_agent()
