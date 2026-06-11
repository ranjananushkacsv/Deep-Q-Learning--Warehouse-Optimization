# Deep-Q-Learning Warehouse-Optimization

A Reinforcement Learning case study where an AI agent **teaches itself** the optimal route through a simulated warehouse — starting with zero knowledge and ending with a provably optimal routing policy.

**Result: 94% reduction in travel per delivery compared to random routing.**

---

## The Problem

A warehouse robot must carry packages from the **inbound station** to the **shipping dock**, navigating around shelf aisles. Nobody gives it a map. Nobody programs the route. It must learn purely from trial and error.

```
S . . . . . . . . .      S = Start (inbound station)
. . █ . . . . █ █ .      G = Goal (shipping dock)
. . █ . . █ . . . .      █ = Shelves (obstacles)
. . █ . . █ . . . .      . = Open floor
. . █ . . █ . . . .
. . █ . . █ . █ . .
. . . . . █ . █ . .
. . . . . . . . . G
```

## How It Learns 

Imagine a new delivery worker with a notebook:

- Every wasted step costs them a small penalty (**−1**)
- Bumping into a shelf costs more (**−5**)
- Delivering the package earns a big reward (**+25**)

After every move, they jot down: *"from this spot, moving right turned out well."* Over 1,000 practice runs, that notebook becomes a complete routing rulebook.

That notebook is the **Q-table**, and the jotting-down rule is the **Bellman equation**:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s',a') − Q(s,a) ]
```

| Symbol | Name | What it means | Value used |
|--------|------|---------------|-----------|
| α | Learning rate | How fast it updates beliefs (20% new, 80% old) | 0.2 |
| γ | Discount factor | How much it values future rewards | 0.95 |
| ε | Exploration rate | Chance of trying a random move (decays over time) | 1.0 → 0.05 |
| r | Reward | Feedback from the environment | −5 / −1 / +25 |

**Exploration → Exploitation:** The agent starts 100% random (exploring the warehouse) and gradually shifts to trusting its Q-table (exploiting what it learned), keeping a 5% exploration floor.

## Results

Benchmarked over 200 simulated deliveries each:

| Metric | Random routing | Q-Learning agent |
|--------|---------------|------------------|
| Avg steps per delivery | 266 | **16** |
| Delivery success rate | 29.5% | **100%** |
| Travel reduction | — | **94%** |

The agent converged to the mathematically optimal 16-step route — and learned the correct move from *every* cell in the warehouse, not just along one path.

## Generated Visualizations

Running the script produces four charts:

1. **`training_curve.png`** — reward per episode climbing from −180 to +8.4 (the learning happening)
2. **`learned_policy.png`** — arrow map showing the best move from every cell
3. **`qvalue_heatmap.png`** — how valuable the agent considers each warehouse position
4. **`kpi_comparison.png`** — before/after bar charts (the business case)

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/warehouse-qlearning.git
cd warehouse-qlearning

# 2. Install dependencies
pip install numpy matplotlib

# 3. Train the agent (~10 seconds)
python warehouse_qlearning.py
```

You'll see live training logs and the four PNG charts will be saved in the project folder.

## How the Code Is Organized

| Section | What it does |
|---------|-------------|
| 1. Environment | Grid world, shelves, reward function (`env_step`) |
| 2. Q-table | 3D NumPy array: `Q[row][col][action]`, initialized to zeros |
| 3. Hyperparameters | α, γ, ε and their decay schedule |
| 4. Training loop | ε-greedy action selection + Bellman update, 1,000 episodes |
| 5. Policy extraction | `argmax` over actions → final routing rulebook |
| 6. Baseline comparison | Random vs trained agent over 200 deliveries |
| 7. Visualization | Generates the four charts with Matplotlib |

## How This Relates to Real Warehouses

Real robotic fulfillment systems (Amazon Robotics, Ocado, GreyOrange) use the same core ideas, scaled up:

- **Bigger state spaces** → the Q-table is replaced by a neural network (Deep Q-Networks)
- **Richer states** → robot position + load status + battery + pending orders
- **Multiple robots** → multi-agent RL with traffic coordination
- **Train in simulation first** → policies are learned in a digital twin (exactly like this project) before touching real hardware

This project is a faithful miniature of that pipeline.

## Possible Extensions

- Multiple pickup points (multi-goal routing)
- Multiple robots with collision avoidance (multi-agent RL)
- Dynamic obstacles (moving forklifts)
- Replace the Q-table with a Deep Q-Network (PyTorch)
- Compare against classical pathfinding (A*, Dijkstra)

## Key Concepts Demonstrated

`Reinforcement Learning` · `Q-Learning` · `Bellman Equation` · `ε-Greedy Exploration` · `Reward Shaping` · `Policy Extraction` · `Process Optimization`

---

*Built with Python, NumPy, and Matplotlib. No ML frameworks required — the entire learning algorithm is one equation.*
