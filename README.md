
# 🧠 Trauma in Reinforcement Learning (RL) with A2C in Breakout

This project explores the **effect of traumatic events on the behavior of an RL agent**, using the game **Breakout** and the **A2C (Advantage Actor-Critic)** algorithm as the training foundation.

## 🎯 Objective

Simulate how a **single traumatic event** (a strong penalty for breaking a block) can affect the long-term behavior of an agent. We’ll compare two trained agents:

- 🟢 **Non-traumatized agent**: Plays normally.  
- 🔴 **Traumatized agent**: Receives a large penalty for breaking a block in the first quarter of the episode.

We aim to observe:

- How the traumatized agent's behavior changes.  
- Whether it avoids the punished action.  
- How much it recovers over time.

---

## 🧪 Why this experiment?

Inspired by human phenomena like trauma and aversive learning, this experiment aims to assess how **a single significant negative event** can alter future decisions in RL agents — especially in contexts where reward maximization is expected.

---

## ⚙️ Code Structure

| File                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `a2c_breakout_trauma.py` | Main script that trains both agents (with and without trauma) and saves models. |
| `view_agents.py`         | Visualizes the behavior of the trained agents.                              |
| `trauma_wrapper.py`      | Defines the wrapper that applies trauma to the agent.                       |

---

## ▶️ How to Run

### 1. Train both agents
```bash
python a2c_breakout_trauma.py
```
This will train both agents (with and without trauma) and save the models.


### 2. Visualize the agents
```bash
python view_agents.py --specific N

```
Displays episode number N for both agents.


### 3. watch all episodes
```bash
python view_agents.py

```
Plays all saved episodes of each agent.

## 📊 What do we analyze?

- **Post-trauma behavior:** Does the agent avoid breaking blocks early after being punished?  
- **Recovery ability:** Does the agent try the punished action again if it sees long-term benefit?  
- **Qualitative comparison:** Video episodes allow visual observation of strategic differences.  
- **Algorithm's role:** A2C is on-policy, which makes it more prone to avoid punished paths — even if they are optimal.

---

## 🧠 Expected Conclusions

- **Trauma introduces a bias** in the agent's learning process, especially if it occurs in early training stages.  
- **On-policy agents like A2C** tend to become more conservative after harsh punishments, avoiding paths that might be optimal.  
- The experiment aims to show that even **a single negative experience** can have a lasting and observable impact.  
- This opens up discussions on parallels with human learning, where traumatic events can strongly shape future behavior.

---

## 📦 Requirements

- Python 3.9+  
- `gymnasium[atari]`  
- `stable-baselines3`  
- `matplotlib`  
- `opencv-python` (if using video recording with OpenCV)

Recommended installation:

```bash
pip install stable-baselines3[extra] gymnasium[atari] matplotlib opencv-python
```

