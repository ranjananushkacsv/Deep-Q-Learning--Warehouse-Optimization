
import numpy as np                      # fast array math for the Q-table
import random                           # for epsilon-greedy coin flips
import matplotlib
matplotlib.use("Agg")                   # render plots to files (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Reproducibility: same random seed = same results every run.
random.seed(42)
np.random.seed(42)

# SECTION 1 — THE WAREHOUSE ENVIRONMENT


GRID_ROWS = 8                           # warehouse is 8 cells deep
GRID_COLS = 10                          # and 10 cells wide

START = (0, 0)                          # inbound station (top-left corner)
GOAL = (7, 9)                           # shipping dock (bottom-right corner)

# Shelving racks the robot cannot drive through.
# Laid out as two long aisles — a realistic warehouse pattern.
SHELVES = {
    (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),         # rack aisle 1 (vertical)
    (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),         # rack aisle 2 (vertical)
    (1, 7), (1, 8),                                  # corner rack near dock
    (5, 7), (6, 7),                                  # rack blocking shortcut
}

# The 4 possible moves, as (row_change, col_change) pairs.
# Index matters: action 0 = up, 1 = down, 2 = left, 3 = right.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]
ACTION_ARROWS = ["↑", "↓", "←", "→"]
NUM_ACTIONS = len(ACTIONS)              # = 4


def env_step(state, action):
    """
    The physics of our warehouse. Given where the robot is (state) and
    what it tries to do (action), return:
        next_state : where it ends up
        reward     : the feedback signal
        done       : True if the package reached the dock

    REWARD DESIGN (this is the heart of "process optimization"):
        -1   per move        -> punishes long routes (time = money)
        -5   hitting a shelf -> punishes collisions / blocked paths
        +25  reaching dock   -> the payoff for completing the delivery
    """
    row, col = state                    # unpack current position
    d_row, d_col = ACTIONS[action]      # unpack the chosen move
    new_row, new_col = row + d_row, col + d_col   # tentative new position

    # Case 1: robot tried to drive off the floor or into a shelf.
    # It stays in place and gets a -5 penalty (a "bump").
    if (not (0 <= new_row < GRID_ROWS and 0 <= new_col < GRID_COLS)
            or (new_row, new_col) in SHELVES):
        return (row, col), -5, False    # stayed put, penalized, not done

    # Case 2: robot reached the shipping dock. Big reward, episode over.
    if (new_row, new_col) == GOAL:
        return (new_row, new_col), +25, True

    # Case 3: a normal move. Small time penalty keeps it motivated to hurry.
    return (new_row, new_col), -1, False

# SECTION 2 — THE Q-TABLE (THE AGENT'S "NOTEBOOK")


Q = np.zeros((GRID_ROWS, GRID_COLS, NUM_ACTIONS))

# SECTION 3 — HYPERPARAMETERS (THE TRAINING KNOBS)

ALPHA = 0.2         # learning rate: how big each notebook update is, 0.2 = blend 20% new experience with 80% old belief.

GAMMA = 0.95        # discount factor: how much the agent values the FUTURE, 0.95 means a reward 10 steps away is worth 0.95^10 ≈ 60% of an immediate reward. High gamma = long-term planner.

EPSILON = 1.0       # exploration rate: probability of a RANDOM move.Starts at 1.0 (100% random — pure exploration).

EPSILON_MIN = 0.05  # never stop exploring entirely (5% floor) — the world could change, so always stay a little curious.

EPSILON_DECAY = 0.995   # after each episode: epsilon *= 0.995. Slowly shifts from exploring to exploiting.
                        

NUM_EPISODES = 1000     # how many practice deliveries the robot makes.
MAX_STEPS = 300         # safety cap so a lost robot doesn't loop forever.

# SECTION 4 — THE TRAINING LOOP (WHERE LEARNING HAPPENS)

episode_rewards = []                    
episode_lengths = []                    
for episode in range(NUM_EPISODES):

    state = START                       
    total_reward = 0                    

    for step_num in range(MAX_STEPS):

        row, col = state

        #EPSILON-GREEDY ACTION SELECTION 
        if random.random() < EPSILON:
            action = random.randint(0, NUM_ACTIONS - 1)     # explore
        else:
            action = int(np.argmax(Q[row, col]))            # exploit

        # TAKE THE ACTION, OBSERVE THE OUTCOME 
        next_state, reward, done = env_step(state, action)
        next_row, next_col = next_state

        #THE BELLMAN UPDATE (the one line that IS Q-learning)

        best_future = np.max(Q[next_row, next_col])          # max_a' Q(s',a')
        target = reward + GAMMA * best_future               # r + γ·max Q(s',a')
        td_error = target - Q[row, col, action]             # surprise amount
        Q[row, col, action] += ALPHA * td_error             # learn!

        state = next_state
        total_reward += reward
        if done:                        # package delivered — episode over
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(step_num + 1)

    # Decay exploration: each episode the robot trusts its notebook more.
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # Progress log every 100 episodes
    if (episode + 1) % 100 == 0:
        recent_avg = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1:4d} | avg reward (last 100): "
              f"{recent_avg:7.1f} | epsilon: {EPSILON:.3f}")

print("\nTraining complete!")

# SECTION 5 — EXTRACT THE LEARNED POLICY


policy = np.argmax(Q, axis=2)           # shape (rows, cols): best action index
state_values = np.max(Q, axis=2)        # shape (rows, cols): value of each cell

# SECTION 6 — BASELINE COMPARISON (THE BUSINESS CASE)


def run_delivery(use_policy):
    """Simulate one delivery. Returns (steps_taken, total_reward, success)."""
    state = START
    total = 0
    for step_num in range(MAX_STEPS):
        row, col = state
        if use_policy:
            action = int(policy[row, col])              # trained brain
        else:
            action = random.randint(0, NUM_ACTIONS - 1) # random wandering
        state, reward, done = env_step(state, action)
        total += reward
        if done:
            return step_num + 1, total, True
    return MAX_STEPS, total, False      # ran out of time = failed delivery


N_TRIALS = 200
random_results = [run_delivery(use_policy=False) for _ in range(N_TRIALS)]
trained_results = [run_delivery(use_policy=True) for _ in range(N_TRIALS)]

rand_steps = np.mean([r[0] for r in random_results])
trained_steps = np.mean([r[0] for r in trained_results])
rand_success = 100 * np.mean([r[2] for r in random_results])
trained_success = 100 * np.mean([r[2] for r in trained_results])

print(f"\n{'='*60}")
print(f"KPI COMPARISON ({N_TRIALS} deliveries each)")
print(f"{'='*60}")
print(f"{'Metric':<30}{'Random':>12}{'Q-Learning':>15}")
print(f"{'-'*60}")
print(f"{'Avg steps per delivery':<30}{rand_steps:>12.1f}{trained_steps:>15.1f}")
print(f"{'Delivery success rate':<30}{rand_success:>11.1f}%{trained_success:>14.1f}%")
improvement = 100 * (rand_steps - trained_steps) / rand_steps
print(f"\n>>> Travel reduced by {improvement:.0f}% — that's the optimization. <<<")

# 
# SECTION 7 — VISUALIZATIONS 


plt.style.use("default")

# ---- 7a. Training curve -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
window = 25                              # smooth with a moving average
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
ax.plot(episode_rewards, alpha=0.25, color="#378ADD", label="Per episode")
ax.plot(range(window-1, NUM_EPISODES), smoothed, color="#185FA5",
        linewidth=2.5, label=f"{window}-episode moving average")
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.set_title("The robot learns: reward per training episode")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("training_curve.png", dpi=150)
plt.close(fig)

# ---- 7b. Learned policy (arrow map) ----------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if (r, c) in SHELVES:
            ax.add_patch(plt.Rectangle((c, GRID_ROWS-1-r), 1, 1,
                         color="#444441"))
        elif (r, c) == GOAL:
            ax.add_patch(plt.Rectangle((c, GRID_ROWS-1-r), 1, 1,
                         color="#EF9F27"))
            ax.text(c+0.5, GRID_ROWS-1-r+0.5, "DOCK", ha="center",
                    va="center", fontsize=9, fontweight="bold")
        elif (r, c) == START:
            ax.add_patch(plt.Rectangle((c, GRID_ROWS-1-r), 1, 1,
                         color="#5DCAA5"))
            ax.text(c+0.5, GRID_ROWS-1-r+0.5, "START", ha="center",
                    va="center", fontsize=8, fontweight="bold")
        else:
            ax.text(c+0.5, GRID_ROWS-1-r+0.5, ACTION_ARROWS[policy[r, c]],
                    ha="center", va="center", fontsize=16, color="#185FA5")
ax.set_xlim(0, GRID_COLS); ax.set_ylim(0, GRID_ROWS)
ax.set_xticks(range(GRID_COLS+1)); ax.set_yticks(range(GRID_ROWS+1))
ax.grid(True, alpha=0.4)
ax.set_aspect("equal")
ax.set_title("Learned routing policy: best move from every cell")
fig.tight_layout()
fig.savefig("learned_policy.png", dpi=150)
plt.close(fig)

# ---- 7c. Q-value heatmap ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
display_vals = state_values.copy()
for (r, c) in SHELVES:
    display_vals[r, c] = np.nan          # blank out shelves
im = ax.imshow(display_vals, cmap="viridis")
ax.set_title("State values: how 'good' each warehouse cell is")
fig.colorbar(im, label="Max Q-value")
ax.set_xticks(range(GRID_COLS)); ax.set_yticks(range(GRID_ROWS))
fig.tight_layout()
fig.savefig("qvalue_heatmap.png", dpi=150)
plt.close(fig)

# ---- 7d. KPI comparison bar chart -------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
ax1.bar(["Random\nrouting", "Q-Learning\nagent"], [rand_steps, trained_steps],
        color=["#B4B2A9", "#1D9E75"])
ax1.set_ylabel("Avg steps per delivery")
ax1.set_title("Travel time per delivery")
for i, v in enumerate([rand_steps, trained_steps]):
    ax1.text(i, v + 2, f"{v:.0f}", ha="center", fontweight="bold")

ax2.bar(["Random\nrouting", "Q-Learning\nagent"], [rand_success, trained_success],
        color=["#B4B2A9", "#1D9E75"])
ax2.set_ylabel("Delivery success rate (%)")
ax2.set_title("Deliveries completed within time cap")
ax2.set_ylim(0, 110)
for i, v in enumerate([rand_success, trained_success]):
    ax2.text(i, v + 3, f"{v:.0f}%", ha="center", fontweight="bold")

fig.suptitle(f"Process optimization result: {improvement:.0f}% less travel",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("kpi_comparison.png", dpi=150)
plt.close(fig)

print("\nSaved: training_curve.png, learned_policy.png, "
      "qvalue_heatmap.png, kpi_comparison.png")
