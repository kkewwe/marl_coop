# Emergent Cooperation in Multi-Agent Reinforcement Learning  
### *Effect of Reward Structure on Predator-Prey Coordination*

---

## **Abstract**
This study explores how the balance between individual and shared rewards affects cooperation among agents in a multi-agent reinforcement learning (MARL) setting.  
Using Deep Q-Network (DQN) agents in a predator-prey environment, we varied the reward-mixing coefficient α (0.0–1.0), where α controls how much of each agent’s reward depends on personal versus team success.  
Results show that cooperation emerges most strongly at intermediate values (α ≈ 0.4), where agents balance self-interest with group coordination.  
These findings highlight how partial cooperation—rather than complete altruism or competition—yields optimal group behavior in MARL.

---

## **1. Introduction**

In reinforcement learning (RL), an agent learns to make sequential decisions by interacting with an environment to maximize cumulative rewards.  
When multiple agents coexist, the learning process becomes **multi-agent reinforcement learning (MARL)**.  
In MARL, agents must learn both *what actions to take* and *how to coordinate with others*, creating a complex system of cooperation and competition.

One of the most influential MARL techniques is the **Deep Q-Network (DQN)**, which combines classical Q-learning with deep neural networks to approximate the *Q-function*:
\[
Q(s, a) = \mathbb{E}[R_t + \gamma \max_{a'} Q(s', a')]
\]
This function estimates the expected reward \(R_t\) from taking action \(a\) in state \(s\) and following the optimal policy thereafter.  
DQN allows agents to handle large state spaces (like images or gridworlds) by replacing tabular storage with neural network estimators.

However, when multiple agents learn simultaneously, their environments become *non-stationary*:  
the state transitions depend not only on the environment but also on the changing policies of other agents.  
Thus, learning stability and coordination depend strongly on how **rewards are distributed** across agents.

---

## **2. Methods**

### **2.1 Environment**
We implemented a **7×7 predator-prey gridworld** in which:
- Two predator agents (the learners) pursue one prey.  
- The prey moves randomly at each time step.  
- Each predator observes its position, the prey’s location, and environmental boundaries.

Episodes end when:
- The prey is caught (team success, reward = 1), or  
- The step limit (100) is reached (failure, reward = 0).

Each agent can choose from five discrete actions: *up*, *down*, *left*, *right*, or *stay still*.

---

### **2.2 Reward Function and the Role of α**

Each agent’s reward combines its **individual** and **team** success:
\[
R_i = \alpha \cdot r_i + (1 - \alpha) \cdot R_{\text{team}}
\]

| Symbol | Meaning |
|--------|----------|
| \(r_i\) | Individual reward (1 if the agent personally catches the prey, else 0) |
| \(R_{\text{team}}\) | Shared team reward (1 if *either* agent catches the prey) |
| α | Reward balance parameter (0 = fully cooperative, 1 = fully selfish) |

- **α = 0.0:** Each predator only cares about the *team’s* success.  
- **α = 0.5:** Each predator cares equally about its own and team reward.  
- **α = 1.0:** Each predator only cares about *its own* catch.

Thus, α controls how much agents are encouraged to **compete versus cooperate**.

---

### **2.3 Agent Architecture**
Each predator is a **Deep Q-Network (DQN)** agent:
- Input: the agent’s observation (grid state and prey position).  
- Output: Q-values for each possible action.  
- Learning uses experience replay (a buffer of past transitions) and a *target network* for stability.

Each agent learns independently from its own experience buffer—known as the *Independent Q-Learning* approach in MARL.  
However, coordination is still possible if reward signals encourage shared goals.

---

### **2.4 Experimental Design**
- **Number of agents:** 2 predators  
- **Grid size:** 7×7  
- **Episodes:** 2000 total  
- **Learning algorithm:** DQN with ε-greedy exploration  
- **Reward structure:** Sweep α from 0.0 to 1.0 in increments of 0.01  
- **Metrics:**
  - *Capture rate* – percentage of episodes where prey was caught  
  - *Coordination index* – similarity between agents’ movement patterns  
  - *Reward variance* – variability between agents’ individual returns  

---

## **3. Results**

### **3.1 Capture Rate vs α**

![Capture Rate vs Alpha](runs/alpha_sweep/capture_rate_vs_alpha.png)

The capture rate initially rises sharply as α increases from 0.0 to around 0.4, reaching nearly **100% success**.  
Beyond α ≈ 0.6, performance deteriorates as agents begin to act more competitively, reducing their ability to coordinate.  
This indicates that **a small degree of individual incentive** is beneficial, but excessive self-interest leads to interference and lower success.

---

### **3.2 Coordination Index (α = 0.0)**

![Coord Index alpha0](runs/alpha0.0_seed0/coord_alpha0.0.png)

At α = 0.0 (fully cooperative), coordination is inconsistent, with low index values (0.2–0.3).  
Because both predators receive the same team reward regardless of contribution, neither agent develops distinct behaviors.  
This leads to *credit assignment failure* — agents cannot tell whether their own actions caused success, slowing learning and creating random movement patterns.

---

### **3.3 Coordination Index (α = 0.5)**

![Coord Index alpha05](runs/alpha0.5_seed0/coord_alpha0.5.png)

At α = 0.5, coordination improves noticeably.  
Agents start to exhibit more structured and complementary behaviors—such as moving in parallel or cornering the prey.  
Because each agent receives partial individual feedback, they can better evaluate their contribution while still benefiting from team reward alignment.

---

### **3.4 Coordination Index (α = 1.0)**

![Coord Index alpha1](runs/alpha1.0_seed0/coord_alpha1.0.png)

At α = 1.0 (fully selfish), coordination nearly vanishes.  
Agents now learn purely independent policies, often **racing** toward the prey or blocking each other’s movement.  
Although individual rewards may occasionally be earned, overall capture efficiency drops sharply, consistent with the declining capture rate at high α.

---

## **4. Discussion**

The results demonstrate a **nonlinear relationship** between reward structure and cooperative behavior.

- **Low α (0–0.2):** Rewards are too uniform, preventing differentiation of good versus bad individual actions.  
  Agents learn slowly, remain uncoordinated, and rely on luck for captures.

- **Moderate α (0.3–0.5):** A balance between individual and team reward creates the highest cooperation.  
  Agents develop consistent strategies where one blocks while the other pursues—emergent teamwork driven by shared but informative incentives.

- **High α (0.6–1.0):** Agents become competitive, acting greedily for personal reward.  
  Cooperation breaks down, and team efficiency declines due to conflicting goals.

Interestingly, performance peaks at **α ≈ 0.4**, slightly below the 0.5 midpoint.  
This suggests that in this environment, **team synergy matters slightly more than self-reward**.  
If individual incentives dominate too early, agents stop learning cooperative positioning; if shared rewards dominate, they stop learning initiative.

This pattern mirrors real-world social dynamics:  
cooperation emerges when individuals retain some self-interest but are still influenced by group success.

---

## **5. Relation to Literature**

This finding aligns with previous work in multi-agent reinforcement learning:
- **Leibo et al. (2017)**, *Social Dilemmas in Multi-Agent RL* – showed that shared rewards promote stable cooperation but must preserve individual learning signals.  
- **Lowe et al. (2017)**, *Multi-Agent Actor-Critic for Mixed Cooperation and Competition* – emphasized that credit assignment is central to effective collaboration.  
- **Foerster et al. (2018)**, *Counterfactual Multi-Agent Policy Gradients* – proposed ways to stabilize cooperative learning by giving agents clearer feedback on their individual contributions.

Your α-sweep experiment empirically demonstrates this balance in a simplified predator-prey domain.

---

## **6. Conclusion**

The α-sweep experiment shows that:
- **Fully shared rewards (α = 0)** lead to aimless cooperation with little learning.
- **Fully independent rewards (α = 1)** lead to selfish interference.
- **Intermediate α (~0.4)** produces the best performance, with agents balancing personal initiative and team coordination.

**Key insight:**  
> Cooperation in MARL does not emerge from pure altruism or competition—it arises from *partially shared incentives* that balance accountability and collaboration.

This principle applies broadly across AI, economics, and biology: the most effective systems reward both individual and group success.

---

## **7. Future Work**

Several directions could extend this experiment:
1. **More Agents:** Test with 3–4 predators to study scaling effects and potential formation of group hierarchies.  
2. **Dynamic α Scheduling:** Allow α to adapt over time (e.g., start cooperative, then individualize), modeling real-world social evolution.  
3. **Communication Channels:** Introduce message-passing or signaling between agents to study explicit coordination.  
4. **Heterogeneous Agents:** Give agents distinct speeds or sensing ranges to analyze asymmetric cooperation.  
5. **Transfer Learning:** Train cooperative agents at α = 0.4, then fine-tune at α = 1.0 to see if learned coordination persists.

---

## **8. References**
- Leibo, J. Z., et al. (2017). *Multi-Agent Reinforcement Learning in Sequential Social Dilemmas.* AAMAS.  
- Lowe, R., et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* NeurIPS.  
- Foerster, J., et al. (2018). *Counterfactual Multi-Agent Policy Gradients.* AAAI.  
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature.

---

